import contextlib
import heapq
import importlib
import os
import time
from numbers import Number
from typing import List, Union
import numpy as np
import inspect

from tinydl import cuda
from tinydl.cuda import (
    Device,
    get_device_from_array,
    CpuDevice,
    GpuDevice,
    check_cuda_available,
    get_device,
    get_array_module,
)

float_type = np.float32
half_type = np.float16
short_type = np.int16
int_type = np.int8
long_type = np.longlong

# 设置数字显示的精度
np.set_printoptions(precision=4)
# 抑制小数的科学计数法的显示
np.set_printoptions(suppress=True)


NdArray = Union["np.ndarray", "cuda.ndarray"]

# 可以转换为数组的数据类型
Arrayable = Union[NdArray, Number, List]
# 可以转为Tensor的数据类型
Tensorable = Union["Tensor", Number, NdArray]


def ensure_array(arrayable: Arrayable, dtype=None, device: Device = None) -> NdArray:
    if device is not None:
        xp = device.xp
    else:
        xp = np

    if isinstance(arrayable, (Number, list)):
        # 通过xp对数据类型进行转换，-> NdArray
        return xp.array(arrayable, dtype=dtype)
    # 如果是ndarray类型的
    elif isinstance(arrayable, (np.ndarray, cuda.ndarray)):
        # 如果指定了cuda，那么需要将数据进行转移
        if device is not None and get_array_module(arrayable) != xp:
            return device.transfer(arrayable)
    # 否则创建cpu上的数据
    return arrayable


def ensure_tensor(tensorable: Tensorable, device: Device = None) -> Tensorable:
    """
    保证该对象是Tensor
    """
    if isinstance(tensorable, Tensor):
        return tensorable
    return Tensor(tensorable, device=device)


# 定义一个全局的配置管理器 Config


class Config:
    debug = False
    backprop = True  # 是否需要计算反向传播梯度


# 通过contextmannager完成，它本质上就是为了简化__enter__ __exit__这两个函数的编写
@contextlib.contextmanager
# 这个decorator接受一个generator
# 用yield语句把with ... as var把变量输出出去
# yield 之前就是enter的过程
# finally 就是exit的过程
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def debug_mode():
    return using_config("debug", True)


def no_grad():
    return using_config("backprop", False)


# 编写一个用于反向传播过程中Debug的包装器
class OpWrapper:

    def __init__(self, name, xs, backward=False) -> None:
        """
        name : op的name
        xs   : op的输出

        """
        self.name = f"back_{name}" if backward else name
        self.xs = xs
        self.output = None
        threshold = int(os.getenv("THRESHOLD", 2))
        self.threshold = threshold

    def __enter__(self):
        if Config.debug:
            self.start = time.time()
        return self

    def __exit__(self, *junk):
        if Config.debug:
            end = (time.time() - self.start) * 1000
            if end > self.threshold:
                print(
                    f"{self.name:>20} : {end:>7.2f} ms {str([y.shape for y in self.xs]):>40} "
                    f"{'-> ' + str(self.output.shape) if self.output is not None else ''}"
                )


class Tensor:
    def __init__(
        self,
        data: Arrayable,
        requires_grad: bool = False,
        dtype=None,
        device: Device = None,
    ) -> None:
        """
        最原始的初始化Tensor的方式，通过个可以转为数组的变量即可完成初始化。
        eg: a = Tensor([[0,1,2],[3,4,5]], requireds_grad=True)
        """
        # 通过其他另一个Tensor构造一个tensor共用同一片内存
        if isinstance(data, Tensor):
            if dtype is None:
                dtype = data.dtype
            if device is None:
                device = data.device
            data = data.data

        if device is None:
            device = get_device_from_array(data)

        self._device = device  # 设备信息
        self.creator = None  # 用于bp
        self.generation = 0  # 用于bp

        # 将data转化为内部支持的NdArray类型（数据，数据位置，数据类型）
        self._data = ensure_array(data, dtype, self._device)
        # 存储当前tensor的梯度，是NdArray类型
        self._grad = None

        self.requires_grad = requires_grad
        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data:NdArray)->None:
        self._data = ensure_array(new_data, device=self.device)
        self._grad = None

    @property
    def grad(self):
        return self._grad

    @property
    def device(self):
        return self._device

    @property
    def xp(self):
        device = self.device
        return np if device is None else device.xp

    # 外部经常调用有关数据的一些属性
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def zero_grad(self) -> None:
        self._grad = self.xp.zeros_like(self.data)

    # 对数据类型进行转换，并返回新的tensor
    # TODO:test
    def type(self, dtype):
        self.data = self.data.astype(dtype)
        return self

    def float(self):
        return self.type(np.float32)

    def short(self):
        return self.type(np.int16)

    def int(self):
        return self.type(np.int8)

    def long(self):
        return self.type(np.int64)

    def bool(self):
        return self.type(np.bool)

    def to(self, device):
        # 这里支持多种device的指定方式：str(cuda:id) 、Device等
        device = get_device(device_decs=device)
        # 已经在相同设备
        if get_device_from_array(self._data) == device:
            return self
        # 否则，进行转移 (数据，设备，梯度)
        self._data = device.transfer(self.data)
        self._device = device

        if self._grad is not None:
            self._grad = Tensor(device.transfer(self._grad), device=device)

        return self

    def to_cpu(self):
        return self.to(CpuDevice())

    def to_gpu(self, device=None):
        check_cuda_available()
        return self.to(get_device(device))

    # 重载一些默认的函数

    def __repr__(self) -> str:
        return (
            f"Tensor(\n{self.data}, requires_grad={self.requires_grad}"
            f"{', device:' + self.device.name if isinstance(self.device, GpuDevice) else ''})"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __gt__(self, other):
        other = ensure_tensor(other, self.device)
        return self.data > other.data

    def __lt__(self, other):
        other = ensure_tensor(other, self.device)
        return self.data < other.data

    def assign(self, x) -> "Tensor":
        x = ensure_tensor(x, self.device)
        assert x.shape == self.shape
        self.data = x.data
        return self

    def size(self, dim=None) -> Union[str, int]:
        """
        如果dim为空，返回shape
        否则，返回dim指定维度上的len
        """
        if dim is None:
            return self.shape

        return self.xp.size(self.data, dim)

    def array(self) -> NdArray:
        return self.data

    def tolist(self):
        return self.data.tolist()

    def item(self) -> Number:
        """将只有一个元素的Tensor转换为Python标量"""
        return self.data.item()

    # 以下是一些用于初始化Tensor的一些常见的分布
    def uniform_(self, low: float = 0.0, high: float = 1.0) -> "Tensor":
        xp = self.xp
        self.data = xp.random.uniform(low, high, size=self.shape).astype(float_type)
        return self

    def normal_(self, mean: float = 0.0, std: float = 1.0) -> "Tensor":
        xp = self.xp
        self.data = xp.random.normal(mean, std, size=self.shape).astype(float_type)
        return self

    # TODO: test
    def index_fill_(self, dim: int, index: "Tensor", value: float) -> "Tensor":
        xp = self.xp
        index = index.data

        # 创建一个和原来的tensor一样大的索引表，但不包括需要修改的维度
        axis = list(range(self.ndim))
        axis.remove(dim)
        # 通过expand_dims将这一维度扩充出来
        index = xp.expand_dims(index, axis=axis)
        # 对这一维度的数据进行填充
        xp.put_along_axis(self.data, index, value, axis=dim)
        return self

    # 除了默认的构造函数之外，其他的便捷的构造函数
    @classmethod
    def empty(cls, *shape, dtype=float_type, device=None, **kwargs):
        device = get_device(device)
        xp = device.xp
        return cls(xp.empty(*shape, dtype=dtype), device=device, **kwargs)

    @classmethod
    def zeros(cls, *shape, dtype=float_type, device=None, **kwargs):
        device = get_device(device)
        xp = device.xp
        return cls(xp.zeros(*shape, dtype=dtype), device=device, **kwargs)

    @classmethod
    def zeros_like(cls, t: "Tensor", **kwargs) -> "Tensor":
        return cls(t.xp.zeros(t.shape, dtype=t.dtype), device=t.device, **kwargs)

    @classmethod
    def ones(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.ones(shape=shape, dtype=dtype), device=device, **kwargs)

    @classmethod
    def ones_like(cls, t: "Tensor", **kwargs) -> "Tensor":
        return cls(t.xp.ones(shape=t.shape, dtype=t.dtype), device=t.device, **kwargs)

    @classmethod
    def randn(cls, *shape, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.random.randn(*shape).astype(dtype), device=device, **kwargs)

    @classmethod
    def arange(
        cls, stop, start=0, step=1, dtype=float_type, device=None, **kwargs
    ) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(
            data=xp.arange(stop=stop, start=start, step=step, dtype=dtype),
            device=device,
            **kwargs,
        )

    @classmethod
    def eye(cls, dim, dtype=float_type, device=None, **kwargs) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.eye(dim).astype(dtype), device=device, **kwargs)

    @classmethod
    def full_like(
        cls,
        t: "Tensor",
        fill_value,
        dtype=float_type,
        requires_grad=False,
        device=None,
        **kwargs,
    ) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(xp.full(t.shape, fill_value), device=device, **kwargs)

    @classmethod
    def uniform(
        cls,
        *shape,
        low: float = -1.0,
        high: float = 1.0,
        dtype=float_type,
        device=None,
        **kwargs,
    ) -> "Tensor":
        device = get_device(device)
        xp = device.xp
        return cls(
            (xp.random.uniform(low, high, size=shape)).astype(dtype),
            device=device,
            **kwargs,
        )

    @classmethod
    def multinomial(cls, input: "Tensor", num_samples: int, replace=False) -> "Tensor":
        """
        返回一个Tensor，每行包含num_samples个索引，从基于input对应行的多项式概率分布采样而来
        Args:
            input: 包含概率的输入，如果不是概率，那么会自动转换为概率
            num_samples: 生成的样本数
            replace: 是否为放回采样，默认为False
        """

        size = input.size(-1)

        assert (
            replace or num_samples <= size
        ), "cannot sample n_sample > input.size(-1) samples without replacement"
        assert input.ndim <= 2, "prob_dist must be 1 or 2 dim"

        p = input.data / input.data.sum(-1, keepdims=True)
        xp = input.xp

        if input.ndim == 1:
            return cls(
                xp.random.choice(
                    xp.arange(size), replace=replace, size=num_samples, p=p
                ),
                device=input.device,
            )
        else:
            ret = []
            for i in range(input.shape[0]):
                ret.append(
                    xp.random.choice(
                        xp.arange(size), replace=replace, size=num_samples, p=p[i]
                    ).tolist()
                )
            return cls(ret, device=input.device)

    # 下面还是重载一些常用的函数

    def __setitem__(self, key, value):
        if isinstance(value, Tensor):
            value = value.data
        self.data[key] = value
        return self

    def __ne__(self, other):
        return Tensor(self.data != ensure_tensor(other, self.device).data)

    # 切片操作
    def __getitem__(self, idxs) -> "Tensor":
        return self.slice(idxs)

    @property
    def T(self) -> "Tensor":
        return self.transpose(axes=None)

    def _get_ops(self, name, *args, **kwargs):
        # 通过动态绑定来找到对应执行的函数，如下面的 repeat, reshape, view等
        # 但是slice,transpose这些函数并不是通过动态绑定完成的
        return self.__getattribute__(name, *args, **kwargs)

    def repeat(self, *sizes):
        if len(sizes) == 1:
            sizes = sizes[0]

        return self._get_ops("_repeat")(sizes)

    def reshape(self, *shape):
        if len(shape) == 1:
            shape = shape[0]

        return self._get_ops("_reshape")(shape)

    def view(self, *shape):
        return self.reshape(shape)

    def permute(self, *shape):
        return self._get_ops("transpose")(shape)

    def expand_dims(self, axis: int):
        return self._get_ops("expanddims")(axis)

    def __array__(self):
        return self.to_cpu().array()

    # detach  创建新的tensor，共享数据，但是不共享计算图，梯度
    def detach(self):
        return Tensor(self)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def unchain(self):
        self.creator = None

    # 反向传播的过程
    def backward(
        self, grad: NdArray = None, retrain_grad=False, create_graph=False
    ) -> None:
        """
        grad        : 这次反向传播的梯度
        retain_grad : 是否保留中间节点的梯度

        creator : 创建Tensor的函数
        funcs   : 函数的堆，通过generation进行排序，generation可以理解为代的意思，就是一个表示创建的前后时间的时间戳。
        反向传播的过程实际上对得到这个Tensor的计算图的反向遍历的过程（也是不严格的拓扑排序），funcs记录的就是计算图中的每一个节点，也就是Function。
        堆是根据每个func的generation来进行排序的，这就保证在计算某个节点时，它之后的节点的梯度已经计算完全。在计算图的遍历过程中，只有某个节点的creator为None说明这个节点是叶子结点。
        """

        assert (
            self.requires_grad
        ), "backward only work on tensor with requires_grad==True"

        if not Config.backprop:
            return

        if grad is None:
            if self.shape == ():
                self._grad = self.xp.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non scalar")
        else:
            self._grad = grad

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, (-f.generation, len(seen_set), f))
                seen_set.add(f)

        add_func(self.creator)
        while funcs:
            _, _, f = heapq.heappop(funcs)
            # 先拿到这个计算图中一个计算节点 f 的所有输出变量的梯度值
            # 才能通过这些梯度再往 f 的输入的变量传播
            gys = [output().grad for output in f.outputs]

            with using_config("backprop", create_graph):
                # OpWrapper 只是为了反向传播过程debug
                with OpWrapper(f.__class__.__name__, gys, backward=True):
                    # gxs就是得到 f 的输入变量的梯度值
                    gxs = f.backward(*gys)

                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                # 现在就是拓扑排序中，找入边的过程，对 f 的每个输入再进行bp
                for x, gx in zip(f.inputs, gxs):
                    if x.requires_grad and gx is not None:
                        assert (
                            x.shape == gx.shape
                        ), f"grad shape must match tensor shape in {f!r}, {gx.shape!r} != {x.shape!r}"

                        if x.grad is None:
                            x._grad = gx
                        else:
                            x._grad = x._grad + gx

                        # 说明 x 还不是整张计算图中的叶子结点
                        if x.creator is not None:
                            add_func(x.creator)

            # 当前这个图节点已经完全计算完成，将梯度进行清除
            if not retrain_grad:
                for y in f.outputs:
                    y()._grad = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()


def register(name, fxn):
    def dispatch(*xs, **kwargs):
        return fxn()(*xs, **kwargs)

    # 为Tensor添加属性，名为name，值为dispatch函数引用

    if name in ["pow", "neg", "abs"]:
        setattr(Tensor, f"__{name}__", dispatch)

    if getattr(Tensor, name, None) is None:
        setattr(Tensor, name, dispatch)
    else:
        setattr(Tensor, f"_{name}", dispatch)

    # 这几个方法都有__xx__, __ixx__, __rxx__ 魔法方法
    if name in ["matmul"]:
        setattr(Tensor, f"__{name}__", dispatch)
        setattr(
            Tensor, f"__i{name}__", lambda self, x: self.assign(dispatch(self, x))
        )  # __i*__ 代表原地操作
        setattr(
            Tensor, f"__r{name}__", lambda self, x: dispatch(x, self)
        )  # __r*__ 代表 other在操作符前, self在操作符后


# 将namespace中的所有Function操作都注册进Tensor
def _register_ops(namespace):
    for name, cls in inspect.getmembers(namespace, inspect.isclass):
        if name[0] != "_" and name != "Tensor":
            # 将Fucntion的子类都进行注册
            register(name.lower(), cls)

try:
    _register_ops(importlib.import_module("tinydl.ops"))

except ImportError as e:
    print(e)


# with this snippet from tinydl/tensor.py:
