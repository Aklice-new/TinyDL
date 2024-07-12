from tinydl.tensor import Tensor # 必须在第一行，先执行_register_ops
import tinydl.functions
import tinydl.init
import tinydl.ops
from tinydl import cuda

tinydl.ops.install_ops()

from tinydl.tensor import no_grad
from tinydl.tensor import ensure_tensor
from tinydl.tensor import ensure_array
from tinydl.tensor import float_type
from tinydl.tensor import debug_mode

from tinydl import module as nn
from tinydl import optim