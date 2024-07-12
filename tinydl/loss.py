from tinydl import functions as F
from tinydl.module import Module
from tinydl.tensor import Tensor


class _Loss(Module):

    reduction: str

    def __init__(self, reduction: str = "mean") -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction


class MSELoss(_Loss):

    def forward(self, intput: Tensor, target: Tensor) -> Tensor:
        errors = (intput - target) ** 2
        if self.reduction == "mean":
            loss = errors.mean()
        elif self.reduction == "sum":
            loss = errors.sum()
        else:
            loss = errors
        return loss


class BCELoss(_Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: logits
        :param target: 真实标签0/1
        """
        return F.binary_cross_entropy(input, target, self.reduction)


class CrossEntropy(_Loss):

    def __init__(self, reduction: str = "mean", ignore_index=-100) -> None:
        super().__init__(reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: logits
        :param target: 真实标签one-hot向量
        """
        return F.cross_entropy(input, target, self.reduction, self.ignore_index)


class NLLLoss(_Loss):
    def __init__(self, reduction: str = "mean", ignore_index=-100) -> None:
        super().__init__(reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input: 对数概率 即 log_softmax
        :param target: 类别索引 或 one-hot向量
        :return:
        """
        return F.nll_loss(input, target, self.reduction, self.ignore_index)
