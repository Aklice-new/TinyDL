import numpy
import tinydl
from numbers import Number
import functools


class _FakeContest:

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


_face_context = _FakeContest()


class Device:
    """
    用于管理Cpu和Gpu的类
    """

    @property
    def xp(self):
        """返回使用的后端计算引擎numpy或者cupy"""
        raise NotImplementedError("Base Class can't be used directly")

    @property
    def name(self):
        """返回设备名称"""
        raise NotImplementedError("Base Class can't be used directly")

    def transfer(self, array):
        """将array的数据转移到该设备上"""
        raise NotImplementedError("Base Class can't be used directly")

    def create_context(self):
        return _face_context

    def __eq__(self, __value: object) -> bool:
        raise NotImplementedError("Base Class can't be used directly")

    def __enter__(self):
        raise NotImplementedError("Base Class can't be used directly")

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError("Base Class can't be used directly")


# 默认是有gpu的，然后后面是判断是否有gpu
gpu_available = True

# 根据是否有cupy包来判断是否有gpu
try:
    import cupy
    import cupyx
    from cupy import cuda, ndarray
    from cupy.cuda import Device as CudaDevice

except ImportError as e:
    print(e)
    # 因为没有导入cupy包，为了保证代码的正常运行，需要手动构造上面导入的一些类
    gpu_available = False

    class ndarray:
        # 只声明一些我们使用的方法（方法名和ndarray中的一致）
        @property
        def shape(self):
            pass

        @property
        def device(self):
            pass

        def get(self):
            pass

        def set(self):
            pass

    class CudaDevice:
        def __init__(self) -> None:
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    cupy = object()


class CpuDevice(Device):
    name = "cpu"
    xp = numpy

    def __repr__(self) -> str:
        return "device(type=cpu)"

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, CpuDevice)

    def transfer(self, array):
        # 防御性编程，如果array是None，那么返回None
        if array is None:
            return None

        # 如果array是numpy.ndarray类型，那么直接返回
        if isinstance(array, numpy.ndarray):
            return array

        # 如果array是数字或者列表，那么转换成numpy.ndarray类型
        if isinstance(array, (Number, list)):
            return numpy.array(array)

        # 如果是标量，那么转换成numpy.ndarray类型
        if numpy.isscalar(array):
            return numpy.array(array)

        # 否则只能是cupy.ndarray类型，那么转换成numpy.ndarray类型
        if isinstance(array, ndarray):
            return array.get()

        raise TypeError(
            f"Unsupported type {type(array)}, can't be converted numpy.ndarray"
        )


class GpuDevice(Device):
    xp = cupy

    def __init__(self, device: CudaDevice) -> None:
        check_cuda_available()

        assert isinstance(device, CudaDevice)
        super(GpuDevice, self).__init__()
        self.device = device

    @property
    def name(self):
        return f"cuda:{self.device.id}"

    # 从device_id来构造
    @staticmethod
    def from_device_id(device_id: int = 0):
        check_cuda_available()

        return GpuDevice(cuda.Device(device_id))

    # 从ndarray来构造
    @staticmethod
    def from_array(array: ndarray):
        if isinstance(array, ndarray) and array.device is not None:
            return GpuDevice(array.device)
        return None

    def create_context(self):
        return cuda.Device(self.device.id)

    def transfer(self, array):
        if array is None:
            return None
        # 首先对数据类型进行转换
        if isinstance(array, (Number, list)):
            array = numpy.array(array)
        # 如果array是cupy.ndarray类型并且就是当前设备的
        if isinstance(array, ndarray):
            if array.device == self.device:
                return array
            # 否则不是numpy类型
            is_numpy = False
        # 判断是numpy类型
        elif isinstance(array, numpy.ndarray):
            is_numpy = True
        else:
            raise TypeError(
                f"Unsupported type {type(array)}, can't be converted numpy.ndarray"
            )
        # 对是numpy的直接通过cupy.asarray转换成cupy类型
        if is_numpy:
            return cupy.asarray(array)
        # 否则通过cupy.array来构造
        return cupy.array(array, copy=True)


# 以下都是一些用于gpu相关信息查询的函数


def is_available():
    return gpu_available


def check_cuda_available():
    if not gpu_available:
        raise RuntimeError("Promise that you have install cupy already!")


def get_device(device_decs) -> Device:
    """
    根据device_desc获取设备(_Deivce)
    Args:
        device_desc: GpuDevice或CpuDevice
                     cpu -> CPU
                     cuda -> 默认显卡
                     cuda:1 -> 指定显卡1

    """
    if device_decs is None:
        return CpuDevice()

    if isinstance(device_decs, Device):
        return device_decs

    if is_available() and isinstance(device_decs, CudaDevice):
        return GpuDevice(device_decs)

    if device_decs == "cpu":
        return CpuDevice()

    if device_decs.startswith("cuda"):
        name, colon, device_id = device_decs.partition(":")
        if not colon:
            device_id = 0
        return GpuDevice.from_device_id(device_id)
    raise ValueError("Invalid argument")


def using_device(device_desc):
    """
    返回当前设备的上下文管理器
    """
    device = get_device(device_desc)
    return device.create_context()


def get_device_from_array(array):
    """
    从array中获取设备
    """
    device = GpuDevice.from_array(array)
    if device is not None:
        return device
    return CpuDevice()


def get_gpu_device_or_current(device):
    """
    获取当前设备，根据device来判断
    """
    check_cuda_available()

    if device is None:
        return cuda.Device()

    if isinstance(device, CudaDevice):
        return device

    if isinstance(device, int):
        return cuda.Device(device)

    raise ValueError("Invalid argument, only support `cuda.Device` or non-negative int")


def get_array_module(array):
    """
    获取array计算引擎是cupy还是numpy
    """
    if is_available():
        if isinstance(array, tinydl.Tensor):
            array = array.data
        return cupy.get_array_module(array)
    return numpy


def memoize(bool_for_each_device=False):
    """
    返回一个装饰器，用于缓存函数的返回值
    """
    if gpu_available:
        return cupy.memoize(bool_for_each_device)

    def dummy_decorator(func):
        @functools.warps(func)
        def ret(*args, **kwargs):
            return func(*args, **kwargs)

        return ret

    return dummy_decorator


def clear_memo():
    if gpu_available:
        cupy.clear_memo()



@memoize()
def elementwise(in_params, out_params, operation, name, **kwargs):
    '''
        调用cupy的ElementwiseKernel去加速GPU运行，注意需要编写C++代码，见 https://docs.cupy.dev/en/stable/user_guide/kernel.html

    Args:
        in_params: 输入参数
        out_params: 输出参数
        operation: 操作
        name: 名称
        **kwargs:

    Returns:

    '''

    check_cuda_available()
    return cupy.ElementwiseKernel(
        in_params, out_params, operation, name, **kwargs)