from typing import Union
from tinydl.tensor import Tensor, Arrayable


class Parameter(Tensor):
    def __init__(self, data: Union[Arrayable, Tensor], dtype=None, device=None) -> None:
        super().__init__(data, requires_grad=True, dtype=dtype, device=device)
