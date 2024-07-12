from typing import Tuple, Union, List
import numpy as np

from tinydl import cuda
from tinydl.cuda import get_array_module
from tinydl.tensor import Tensor, NdArray
from tinydl.ops import Function


class Embedding(Function):
    def forward(self, weight: NdArray, indices: NdArray) -> NdArray:
        self.save_for_backward(weight.shape, indices)
        return weight[indices]

    def backward(self, grad: NdArray) -> Tuple[NdArray, None]:
        w_shape, indices = self.saved_tensors

        xp = get_array_module(grad)

        bigger_grad = xp.zeros(w_shape, dtype=grad.dtype)

        if xp is np:
            np.add.at(bigger_grad, indices, grad)
        else:
            bigger_grad.scatter_add(indices, grad)

        return bigger_grad, None


def embedding(weight: Tensor, indices: Tensor) -> Tensor:
    return Embedding()(weight, indices)


class Dropout(Function):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        if xp is np:
            scale = x.dtype.type(1.0 / (1 - self.p))
            flag = np.random.rand(*x.shape) >= self.p
            mask = scale * flag
            y = x * mask
        else:
            rand = xp.random.rand(*x.shape, dtype=np.float32)
            scale = x.dtype.type(1.0 / (1 - self.p))
            mask, y = cuda.elementwise(
                "T x, R r, T scale, T p",
                "T mask, T y",
                """
                mask = (r >= p) * scale;
                y = x * mask;
                """,
                "dropout_fwd",
            )(x, rand, scale, self.p)
        self.save_for_backward(mask)
        return y

    def backward(self, grad: NdArray) -> NdArray:
        (mask,) = self.saved_tensors
        return mask * grad


def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if training:
        return Dropout(p)(x)
    else:
        return x


class ReLU(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        y = xp.maximum(x, 0, dtype=x.dtype)
        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        (y, xp) = self.saved_tensors
        if xp is np:
            return grad * (y > 0)
        else:
            return cuda.elementwise(
                "T y, T gy", "T gx", "gx = y > 0 ? gy : (T)0", "relu_bwd"
            )(y, grad)


def relu(x: Tensor) -> Tensor:
    return ReLU()(x)


class Sigmoid(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)

        if xp is np:
            half = x.dtype.type(0.5)
            y = np.tanh(x * half) * half + half
        else:
            y = cuda.elementwise(
                "T x", "T y", "y = tanh(x * 0.5) * 0.5 + 0.5", "sigmoid_fwd"
            )(x)

        self.save_for_backward(y, xp)

        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        if xp is np:
            one = y.dtype.type(1)
            return grad * y * (one - y)
        else:
            return cuda.elementwise(
                "T y, T gy", "T gx", "gx = gy * y * (1 - y)", "sigmoid_bwd"
            )(y, grad)


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid()(x)


def logsigmoid(x: Tensor) -> Tensor:
    return sigmoid(x).log()


# 损失相关的函数


def _reduction(errors: Tensor, method: str) -> Tensor:
    if method == "mean":
        loss = errors.sum() / errors.shape[0]
    elif method == "sum":
        loss = errors.sum()
    else:
        loss = errors

    return loss


def _logsumexp(x: NdArray, axis=-1):
    """
    max + \log(\sum (\exp(z_i - max)) )
    """
    xp = get_array_module(x)
    b = x.max(axis=axis, keepdims=True)
    y = x - b
    xp.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    xp.log(s, out=s)
    b += s
    return b


class LogSoftmax(Function):
    """
    max_val = max(x) 减去最大值，防止指数溢出

    \log \frac{\exp(x_i - max_val)}{\sum \exp(x_j - max_val)}

    """

    def __init__(self, axis=-1) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        y = x - _logsumexp(x, self.axis)
        self.save_for_backward(y, xp)
        return y

    def backward(self, grad: NdArray) -> NdArray:
        y, xp = self.saved_tensors
        return grad - xp.exp(y) * grad.sum(axis=self.axis, keepdims=True)


def log_softmax(x: Tensor, axis=-1):
    return LogSoftmax(axis=axis)(x)


class NLLLoss(Function):
    def __init__(self, ignore_index=-100, reduction: str = "mean") -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: NdArray, target: NdArray) -> NdArray:
        """
        args:
            input : 对数概率(log_softmax), shape:(batch_size, num_classes)
            target: 类别索引或者one-hot,   shape:(batch_size,) (batch_size, num_classes)
        """
        xp = get_array_module(input)

        # 如果target是one-hot编码的向量，转为一维向量
        if target.ndim > 1:
            target = xp.argmax(target, axis=1)

        batch_size, num_classes = input.shape

        mask = (target != self.ignore_index).astype(int)

        errors = -xp.sum(input[xp.arange(batch_size), target] * mask, dtype=input.dtype)

        if self.reduction == "mean":
            errors = xp.divide(errors, mask.sum(), dtype=input.dtype)

        self.save_for_backward(target, input, batch_size, num_classes, mask)
        return errors

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        target, input, batch_size, num_classes, mask = self.saved_tensors

        if target.ndim > 1:
            target = xp.argmax(target, axis=1)

        bigger_grad = xp.zeros((batch_size, num_classes), dtype=grad.dtype)
        bigger_grad[xp.arange(batch_size), target] = xp.divide(
            -mask, mask.sum(), dtype=input.dtype
        )
        return bigger_grad


def nll_loss(input: Tensor, target: Tensor, reduction: str = "mean", ignore_index=-100):
    return NLLLoss(ignore_index, reduction)(input, target)


def binary_cross_entropy(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    """
    :param input: logits
    :param target: 真实标签 0或1
    :param reduction:
    :return: binary cross entropy loss
    """
    neg_abs = -abs(input)
    errors = input.clip(x_min=0) - input * target + (1 + neg_abs.exp()).log()

    return _reduction(errors, reduction)


def cross_entropy(
    input: Tensor, target: Tensor, reduction: str = "mean", ignore_index=-100
) -> Tensor:
    """
    :param input: logits
    :param target: 真实标签one-hot向量 或 类别索引
    :param reduction:
    :return:
    """
    # 先计算logsoftmax
    log_y = log_softmax(input)
    # 基于nll实现交叉熵损失
    return nll_loss(log_y, target, reduction, ignore_index)
