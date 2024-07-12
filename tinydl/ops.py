from typing import Any, List, Tuple, Union
import weakref
import numpy as np

from tinydl.tensor import NdArray, Config, Arrayable,Tensor
from tinydl import cuda
from tinydl.cuda import get_array_module, ndarray



class Function:
    def __init__(self) -> None:
        # 保存需要在backward()中使用的Tensor或其他对象(如Shape)
        self.saved_tensors = []
        self.inputs: List[Tensor] = None
        self.outputs: List[Tensor] = None
        self.generation: int = None

    def save_for_backward(self, *x: Any) -> None:
        self.saved_tensors.extend(x)

    def forward(self, *args: Any, **kwargs: Any) -> NdArray:
        """前向传播，进行真正运算的地方"""
        raise NotImplementedError(
            "You must implement the forward function for custom Function."
        )

    def backward(self, grad: NdArray) -> Any:
        """实现反向传播，计算梯度"""
        raise NotImplementedError(
            "You must implement the backward method for your custom Function "
            "to use it with backward mode AD."
        )

    def __call__(self, *xs: "Tensor", **kwargs) -> "Tensor":
        raw_xs = [x.data if isinstance(x, Tensor) else x for x in xs]
        # [t.data for t in xs]遍历Tensor中的data(NdArray)值，参与实际计算的都是NumPy的数组。
        ys = self.forward(*raw_xs, **kwargs)
        requires_grad = any([t.requires_grad for t in xs if isinstance(t, Tensor)])

        return_tuple = True
        if not isinstance(ys, tuple):
            return_tuple = False
            ys = (ys,)

        outputs = [Tensor(y, requires_grad=requires_grad) for y in ys]

        if Config.backprop:
            self.generation = max([x.generation for x in xs if isinstance(x, Tensor)])
            for output in outputs:  # 设定每个输出是由此函数得到的
                output.set_creator(self)
            self.inputs = xs  # 记录输入
            self.outputs = [
                weakref.ref(output) for output in outputs
            ]  # 通过弱引用保存输出

        # 返回多个则通过元组
        if return_tuple or len(outputs) > 1:
            return tuple(outputs)
        return outputs[0]

    # 一些用于处理由于广播操作导致shape不匹配的函数


"""
广播机制是一种方便处理矩阵向量标量计算的高级操作。其匹配的一般原则如下
    -   1x1的标量可以广播为任何形状
    -   在广播时，维度从后往前匹配，对形状不够的进行填补.例如：shape(2,1,4,5) x shape(1,5) = shape(2,1,4,5)，其中第二个矩阵中被扩充的维度有(2,1)，同时1这个维度被扩充成为了4。

在广播的逆操作时需要注意的有两点：
-  一个是原本没有的维度，直接添加进来的。 直接在这些维度上进行求和，将维度抹平。
-  另一个是原本是1，然后扩充的。         在这些维度上也进行求和，但保持维度1。
"""


def unbroadcast(grad: NdArray, in_shape: Tuple) -> NdArray:
    """
    由于广播操作会导致输出的shape和输入的shape不匹配，所以需要进行逆操作来进行纠正
    args:
        grad : output的梯度
        in_shape : input的shape
    """
    # 计算多扩充的维度
    ndims_added = grad.ndim - len(in_shape)
    # 将多扩充的维度进行消除抹平，（多加的维度都在左边）
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)

    # 在input为1的维度可能会被扩充成大于1的，所以这里也需要做处理
    for i, dim in enumerate(in_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def restore_grad_shape(original_shape: Tuple, ndim: int, axis=None) -> Tuple:
    """
    args:
        original_shape : 原始的shape
        ndim : len(shape)
        axis : Mean,Sum,Squeeze这些方法操作的维度
    """
    if isinstance(axis, (np.ndarray, ndarray)):
        axis = axis.item()

    axis = [axis] if np.isscalar(axis) else axis
    axis = tuple(ax % ndim for ax in axis)

    # 对于原始shape的每一维，如果axis中有这个id，那么把这一维变为1，否则保留原样
    return [s if id not in axis else 1 for id, s in enumerate(original_shape)]


def broadcast_grad_shape(
    original_shape: Tuple, xp, grad: NdArray, axis=None, keepdims=None
) -> NdArray:
    """
    在Mean、Sum、Squeeze等方法中，可能会丢失dim=1的维度，不能直接进行广播，需要调用此方法进行一些处理，广播到原来的维度
     Args:
         original_shape: 原来的形状
         xp: numpy 或 cupy
         grad: 梯度
         axis: None 或 int 或 int元组，操作的维度
         keepdims: 是否保存维度，如果为True，那么就不需要调用此方法了。调用此方法的目的就是得到keepdims=True的结果
     Returns: 转换好形状并广播了的grad
    """
    ndim = len(original_shape)

    # 一般情况不指定axis的话，整个Tensor都会被规约成为一个标量，ndim=0
    # 所以直接将shape进行广播即可
    if ndim == 0 or axis is None or keepdims:
        return np.broadcast_to(grad, original_shape)

    grad = grad.reshape(restore_grad_shape(original_shape, ndim, axis))

    return xp.broadcast_to(grad, original_shape)


#  以下都是Function的子类，在整个计算图中就是每个计算的节点，所以需要实现基本的fw，bw的过程

# 二元运算


class Add(Function):
    """
    Tensor + Tensor
    """

    # 在所有的计算过程中，只对NdArray操作不对Tensor直接操作
    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x.shape, y.shape)
        return xp.add(x, y)

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        shape_x, shape_y = self.saved_tensors
        return unbroadcast(grad, shape_x), unbroadcast(grad, shape_y)


class AddConstant(Function):
    """
    Tensor + Scalar
    """

    def forward(self, x: NdArray, scalar) -> NdArray:
        return x + x.dtype.type(scalar)

    def backward(self, grad: NdArray) -> NdArray:
        return grad


def get_numpy_data(tensor):
    if isinstance(tensor, Tensor):
        return tensor.data
    return tensor


def add(self, rhs):
    if np.isscalar(rhs):
        return AddConstant()(self, rhs)
    return Add()(self, rhs)


def add_(self, other, alpha=1):
    other = get_numpy_data(other)
    self.data = self.data + alpha * other
    return self


class Sub(Function):
    """
    Tensor - Tensor
    """

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        self.save_for_backward(x.shape, y.shape)
        return x - y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        shape_x, shape_y = self.saved_tensors
        return unbroadcast(grad, shape_x), unbroadcast(-grad, shape_y)


def sub(self, rhs):
    if np.isscalar(rhs):
        return AddConstant()(self, -rhs)
    return Sub()(self, rhs)


class SubFromConstant(Function):
    """
    常量 - Tensor
    """

    def forward(self, x: NdArray, constant) -> NdArray:
        return x.dtype.type(constant) - x

    def backward(self, grad: NdArray) -> NdArray:
        return -grad


def rsub(self, rhs):
    if np.isscalar(rhs):
        return SubFromConstant()(self, rhs)
    return Sub()(rhs, self)


class Mul(Function):
    """
    Tensor*Tensor
    """

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x, y)
        return xp.multiply(x, y)

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        xp = get_array_module(grad)
        x, y = self.saved_tensors
        return unbroadcast(xp.multiply(grad, y), x.shape), unbroadcast(
            xp.multiply(grad, x), y.shape
        )


class MulConstant(Function):
    """
    Tensor * Scalar
    """

    def forward(self, x: NdArray, scalar) -> NdArray:
        self.save_for_backward(scalar)
        return x * scalar

    def backward(self, grad: NdArray) -> NdArray:
        (scalar,) = self.saved_tensors
        return grad * scalar


def mul(self, rhs):
    if np.isscalar(rhs):
        return MulConstant()(self, rhs)
    return Mul()(self, rhs)


def mul_(self, other):
    other = get_numpy_data(other)
    self.data = self.data * other
    return self


def addcmul_(self, other1, other2, value=1):
    """
    self.data += value*other1*other2
    """
    other1 = get_numpy_data(other1)
    other2 = get_numpy_data(other2)

    self.data = self.data + value * other1 * other2
    return self


def addcdiv_(self, other1, other2, value=1):
    """
    self.data += value * other1 / other2
    """
    other1 = get_numpy_data(other1)
    other2 = get_numpy_data(other2)

    self.data = self.data + value * other1 / other2
    return self


class TrueDiv(Function):
    """
    真除法 保留小数
    Tensor / Tensor
    """

    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        self.save_for_backward(x, y)
        return x / y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        xp = get_array_module(grad)
        x, y = self.saved_tensors

        if xp is np:
            gx = grad / y
            gy = -gx * x / y
        else:
            # 使用自定义内核代码加速
            gx, gy = cuda.elementwise(
                "T x, T y, T grad",
                "T gx, T gy",
                """
                gx = grad / y;
                gy = -gx * x / y;
                """,
                "div_bwd",
            )(x, y, grad)
        return unbroadcast(gx, x.shape), unbroadcast(gy, y.shape)


def div(self, rhs):
    if np.isscalar(rhs):
        return MulConstant()(self, 1.0 / rhs)
    return TrueDiv()(self, rhs)


class DivFromConstant(Function):
    """
    Scalar / Tensor
    """

    def forward(self, x: NdArray, scalar) -> NdArray:
        self.save_for_backward(x, scalar)
        return scalar / x

    def backward(self, grad: NdArray) -> NdArray:
        x, c = self.saved_tensors
        xp = get_array_module(grad)
        if xp is np:
            gx = -c * grad / (x**2)
        else:
            # 使用自定义内核代码加速
            gx = cuda.elementwise(
                "T x, T y, T grad",
                "T gx",
                """
                gx = -y * grad / (x*x);
                """,
                "div_from_const_bwd",
            )(x, c, grad)
        return gx


def rdiv(self, rhs):
    if np.isscalar(rhs):
        return DivFromConstant()(self, rhs)
    return TrueDiv()(rhs, self)


# 规约运算


class Sum(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        self.save_for_backward(x.shape, axis, keepdims)
        return x.sum(axis, keepdims=keepdims)

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        x_shape, axis, keepdims = self.saved_tensors
        return broadcast_grad_shape(x_shape, xp, grad, axis, keepdims)


class Mean(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        out = x.mean(axis, keepdims=keepdims)
        self.save_for_backward(x.shape, out.shape, axis, keepdims)
        return out

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        x_shape, out_shape, axis, keepdims = self.saved_tensors
        grad = grad * (np.prod(out_shape) / np.prod(x_shape))

        return broadcast_grad_shape(x_shape, xp, grad, axis, keepdims)


class Max(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        xp = get_array_module(x)
        y = xp.amax(x, axis=axis, keepdims=keepdims)
        self.save_for_backward(x, axis, y, keepdims)
        return y

    def backward(self, grad: NdArray) -> NdArray:
        x, axis, y, keepdims = self.saved_tensors

        if axis is None:
            mask = x == y
            div = mask.sum(axis=axis, keepdims=keepdims)
        else:
            shape = restore_grad_shape(x.shape, x.ndim, axis)
            grad = grad.reshape(shape)
            y = y.reshape(shape)

            mask = x == y
            div = mask.sum(axis=axis, keepdims=True)
        return mask * grad / div


class Min(Function):
    def forward(self, x: NdArray, axis=None, keepdims=False) -> NdArray:
        xp = get_array_module(x)
        y = xp.amin(x, axis=axis, keepdims=keepdims)
        self.save_for_backward(x, axis, y, keepdims)
        return y

    def backward(self, grad: NdArray) -> NdArray:
        x, axis, y, keepdims = self.saved_tensors
        if axis is None:

            div = mask.sum(axis=axis, keepdims=keepdims)
        else:
            shape = restore_grad_shape(x.shape, x.ndim, axis=axis)
            grad = grad.reshape(shape)
            y = y.reshape(shape)

            mask = x == y
            div = mask.sum(axis=axis, keepdims=keepdims)
            return grad * mask / div


class Clip(Function):
    def forward(self, x: NdArray, x_min=None, x_max=None) -> NdArray:
        xp = get_array_module(x)
        if x_min is None:
            x_min = xp.min(x)
        if x_max is None:
            x_max = xp.max(x)

        self.save_for_backward(x, x_min, x_max)
        return xp.clip(x, x_min, x_max)

    def backward(self, grad: NdArray) -> NdArray:
        x, x_min, x_max = self.saved_tensors
        mask = (x >= x_min) * (x <= x_max)
        return grad * mask


# TODO
class Gather(Function):
    def forward(self, x: NdArray, axis: int, indices) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x.shape, indices, axis)
        return xp.take_along_axis(x, indices, axis)

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        x_shape, indices, axis = self.saved_tensors
        shape_ones = (1,) * indices.ndim
        dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))
        fancy_index = []
        for dim, n in zip(dest_dims, x_shape):
            if dim is None:
                fancy_index.append(indices)
            else:
                ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1 :]
                fancy_index.append(xp.arange(n).reshape(ind_shape))
        bigger_grad = xp.zers(x_shape, dtype=grad.dtype)
        if xp is np:
            xp.add.at(bigger_grad, tuple(fancy_index), grad)
        else:
            cuda.cupyx.scatter_add(bigger_grad, tuple(fancy_index), grad)

        return bigger_grad


def dim_one(shape):
    """
    返回维度为1的索引
    """
    ret = []
    for i, s in enumerate(shape):
        if s == 1:
            ret.append(i)
    return ret


class Squeeze(Function):
    def forward(self, x: NdArray, axis: Union[None, int, Tuple] = None) -> NdArray:
        xp = get_array_module(x)

        if isinstance(axis, (np.ndarray, ndarray)):
            axis = axis.item()

        self.save_for_backward(x.shape, axis)
        return xp.squeeze(x, axis=axis)

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, axis = self.saved_tensors

        if axis is None:
            axis = tuple(dim_one(x_shape))

        shape = restore_grad_shape(x_shape, len(x_shape), axis)

        return grad.reshape(shape)


class Unsqueeze(Function):
    def forward(self, x: NdArray, axis: int) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x.shape)
        if isinstance(axis, NdArray):
            axis = axis.item()

        return xp.expand_dims(x, axis)

    def backward(self, grad: NdArray) -> NdArray:
        (x_shape,) = self.saved_tensors
        return grad.reshape(x_shape)


# 矩阵运算


class Matmul(Function):
    def forward(self, x: NdArray, y: NdArray) -> NdArray:
        assert (
            x.ndim > 1 and y.ndim > 1
        ), f"the dim number of x or y must >=2, actual x:{x.ndim}  and y:{y.ndim}"
        self.save_for_backward(x, y)
        return x @ y

    def backward(self, grad: NdArray) -> Tuple[NdArray, NdArray]:
        x, y = self.saved_tensors
        return unbroadcast(grad @ y.swapaxes(-2, -1), x.shape), unbroadcast(
            x.swapaxes(-2, -1) @ grad, y.shape
        )


# 一元运算


class Pow(Function):
    def forward(self, x: NdArray, c: float) -> NdArray:
        self.save_for_backward(x, c)
        return x**c

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        x, c = self.saved_tensors
        if xp is np:
            return c * x ** (c - 1) * grad
        else:
            return cuda.elementwise(
                "T x, T grad, T c", "T gx", "gx = c * pow(x, c - 1) * grad", "pow_bwd"
            )(x, grad, c)


class Log(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x)
        return xp.log(x)

    def backward(self, grad: NdArray) -> NdArray:
        (x,) = self.saved_tensors
        return grad / x


class Exp(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        ret = xp.exp(x)
        self.save_for_backward(ret)
        return ret

    def backward(self, grad: NdArray) -> NdArray:
        (ret,) = self.saved_tensors
        return grad * ret


class Neg(Function):
    def forward(self, x: NdArray) -> NdArray:
        return -x

    def backward(self, grad: NdArray) -> NdArray:
        return -grad


class Abs(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x)
        return xp.abs(x)

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        (x,) = self.saved_tensors
        if xp is np:
            return grad * xp.sign(x)
        else:
            return cuda.elementwise(
                "T x, T gy", "T gx", "gx = ((x > 0) - (x < 0)) * gy", "abs_bwd"
            )(x, grad)


class Sqrt(Function):
    def forward(self, x: NdArray) -> NdArray:
        xp = get_array_module(x)
        ret = xp.sqrt(x)
        self.save_for_backward(ret)
        return xp.sqrt(ret)

    def backward(self, grad: NdArray) -> NdArray:
        (ret,) = self.saved_tensors
        return grad / (ret * 2.0)


# 形状操作相关函数


class Slice(Function):
    def forward(self, x: NdArray, slices: Any) -> NdArray:
        self.save_for_backward(x.shape, slices)
        return x[slices]

    def backward(self, grad: NdArray) -> NdArray:
        x_shape, slices = self.saved_tensors
        xp = get_array_module(grad)
        bigger_grad = xp.zeros(x_shape, dtype=grad.dtype)

        if xp is np:
            np.add.at(bigger_grad, slice, grad)
        else:
            bigger_grad.scatter_add(slices, grad)
        return bigger_grad


class Sort(Function):
    def forward(self, x: NdArray, axis=-1, descending=False) -> Tuple[NdArray, NdArray]:
        xp = get_array_module(x)

        sorted_indices = xp.argsort(x, axis=axis)
        sorted_x = xp.take_along_axis(x, sorted_indices, axis=axis)

        if descending:
            sorted_x = xp.flip(sorted_x, axis=axis)
            sorted_indices = xp.filp(sorted_indices, axis=axis)

        self.save_for_backward(axis, sorted_indices)

        return sorted_x, sorted_indices

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        axis, indices = self.saved_tensors
        inverse_permutation = xp.argsort(indices, axis=axis)

        return xp.take_along_axis(grad, inverse_permutation, axis=axis)


class Reshape(Function):
    def forward(self, x: NdArray, shape: Tuple) -> NdArray:
        self.save_for_backward(shape)
        return x.reshape(shape)

    def backward(self, grad: NdArray) -> NdArray:
        (x_shape,) = self.saved_tensors
        return grad.reshape(x_shape)


class ExpandDims(Function):
    def forward(self, x: NdArray, axis: int) -> NdArray:
        xp = get_array_module(x)
        self.save_for_backward(x.shape)
        return xp.expand_dims(x, axis=axis)

    def backward(self, grad: NdArray) -> NdArray:
        (x_shape,) = self.saved_tensors
        return grad.reshape(x_shape)


class Transpose(Function):
    def forward(self, x: NdArray, axes) -> NdArray:
        self.save_for_backward(axes)
        return x.transpose(axes)

    def backward(self, grad: NdArray) -> NdArray:
        (axes,) = self.saved_tensors
        if axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(axes)))


class Repeat(Function):
    def forward(self, x: NdArray, repeats) -> NdArray:
        xp = get_array_module(x)
        if isinstance(repeats, int):
            repeats = xp.array(
                repeats,
            )
        elif isinstance(repeats, Tuple):
            repeats = xp.array(repeats)
        self.save_for_backward(x, repeats)
        return xp.tile(x, repeats.tolist())

    def backward(self, grad: NdArray) -> NdArray:
        xp = get_array_module(grad)
        x, repeats = self.saved_tensors

        num_unsqueezed = grad.ndim - x.ndim

        for _ in range(num_unsqueezed):
            grad = grad.sum(0, keepdims=False)

        if repeats.ndim == 0:
            repeats = xp.expand_dims(repeats, 0)

        for dim, repeat in enumerate(repeats[num_unsqueezed]):
            if repeat == 1:
                continue
            grad = sum(xp.array_split(grad, repeat.tolist(), dim))

        return grad


# 有关索引的操作


class _IndexSelect(Function):

    def fwd(self, x: NdArray, xp, axis):
        raise NotImplementedError("You must implement the fwd function in sub class.")

    def forward(self, x: NdArray, axis=None, *args) -> NdArray:
        xp = get_array_module(x)
        return self.fwd(x, xp, axis, *args)

    def backward(self, grad: NdArray) -> NdArray:
        return None


class ArgMax(_IndexSelect):
    def fwd(self, x: NdArray, xp, axis=None):
        return xp.argmax(x, axis=axis)


class ArgMin(_IndexSelect):
    def fwd(self, x: NdArray, xp, axis=None):
        return xp.argmin(x, axis=axis)


# 将基本操作绑定到Tensor的符号中（重载运算符)
# 这里只对 加法，减法，乘法和除法进行了绑定
# 复杂的运算，是通过tensor中的register方法进行绑定的


def install_ops():
    Tensor.__add__ = add
    Tensor.__iadd__ = lambda self, x: self.assign(add(self, x))
    Tensor.__radd__ = add
    Tensor.__sub__ = sub
    Tensor.__rsub__ = rsub
    Tensor.__isub__ = lambda self, x: self.assign(sub(self, x))
    Tensor.__mul__ = mul
    Tensor.__rmul__ = mul
    Tensor.__imul__ = lambda self, x: self.assign(mul(self, x))
    Tensor.__truediv__ = div
    Tensor.__rtruediv__ = rdiv
    Tensor.__itruediv__ = lambda self, x: self.assign(div(self, x))
    Tensor.add_ = add_
    Tensor.mul_ = mul_
    Tensor.addcmul_ = addcmul_
    Tensor.addcdiv_ = addcdiv_
