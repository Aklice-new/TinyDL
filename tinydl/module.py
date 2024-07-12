import math
from collections import OrderedDict
from typing import Optional, List, Set, Tuple, Iterator, Union
import pickle
import operator
from itertools import chain, islice

from tinydl.parameter import Parameter
from tinydl.tensor import Tensor, no_grad, float_type
from tinydl import init
import tinydl.functions as F


def _addindent(s_, numSpaces):
    s = s_.split("\n")
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


class Module:
    """
    所有模型的基类
    """

    training: bool

    def __init__(self) -> None:
        super().__setattr__("training", True)
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_modules", OrderedDict())

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        self._parameters[name] = param

    def add_module(self, name: str, module: Optional["Module"]) -> None:
        self._modules[name] = module

    def get_submodule(self, target: str) -> "Module":
        if target == "":
            return self
        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms:
            mod = getattr(mod, item)

        return mod

    def get_parameter(self, target: str) -> Parameter:
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)
        param: Parameter = getattr(mod, param_name)

        return param

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicae: bool = True,
    ):
        """
        递归的获取所有的子模块
        """
        if memo is None:
            memo = set()

        if self not in memo:
            if remove_duplicae:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicae):
                    yield m

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    def _named_members(self, get_member_fn, prefix="", recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]

        for module_prefix, module in modules:
            members = get_member_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """
        recurse: 是否返回该module下所有的模块的参数
        """
        gen = self._named_members(
            lambda module: module._parameters.items(), prefix=prefix, recurse=recurse
        )
        for elem in gen:
            yield elem

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def children(self) -> Iterator["Module"]:
        for name, module in self.named_children():
            yield module

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.modules():
            module.train(mode)

    def eval(self):
        return self.train(False)

    def state_dict(self, destination=None, prefix=""):
        if destination is None:
            destination = OrderedDict()

        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".")

        return destination

    def _load_from_state_dict(self, state_dict, prefix):
        # 当前模型的参数
        local_name_params = self._parameters.items()
        local_state = {k: v for k, v in local_name_params if v is not None}

        # 将state_dict中的参数都转移到模型中
        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                with no_grad():
                    param.data = input_param

    def load_state_dict(self, state_dict):
        state_dict = OrderedDict(state_dict)

        def load(module, state_dict, prefix=""):
            module._load_from_state_dict(state_dict, prefix)
            for name, child in module._modules.items():
                if child is not None:
                    child_prefix = prefix + name + "."
                    child_state_dict = {
                        k: v
                        for k, v in state_dict.items()
                        if k.startswith(child_prefix)
                    }
                    load(child, child_state_dict, child_prefix)

        load(self, state_dict)
        del load

    def save(self, path="model.pkl"):
        # 保存当前所有模块内部的参数
        state_dict = self.state_dict()
        with open(path, "wb") as f:
            pickle.dump(state_dict, f)
            print(f"Save module to {path}")

    def load(self, path="model.pkl"):
        # 读取模型参数
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for key, param in self._parameters.items():
            if param is None:
                continue
            with no_grad():
                param_applied = fn(param)

            out_param = Parameter(param_applied)
            self._parameters[key] = out_param
        return self

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def to_gpu(self, device):
        return self._apply(lambda t: t.to_gpu(device))

    def to_cpu(self):
        return self._apply(lambda t: t.to_cpu())

    def to(self, device):
        return self._apply(lambda t: t.to_gpu(device))

    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        """
        通过魔法方法，将属性注册到Module中
        args:
            name : 属性名
            value : 具体的属性，这里对Module(子模块)和Tensor(Parameter)进行了特殊的处理
        """

        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        # 注册一个parameter
        if isinstance(value, Parameter):
            # 未被实例化
            if params is None:
                raise AttributeError(
                    "Cannot assign parameters before Module.__init__() call"
                )
            # 如果之前有同名的param将其删除
            remove_from(self.__dict__, self._modules)
            # 将param进行注册
            self.register_parameter(name, value)
        # 不能注册非Parameter类型的Param属性
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{value}'  as parameter '{name}' '(torch.nn.Parameter or None expected)'"
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            # 注册子模块
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(self.__dict__, self._parameters)
                modules[name] = value
            # 不能注册非Module类型的子模块
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign '{value}' as child module '{name}' "
                        "(torch.nn.Module or None expected)"
                    )
                modules[name] = value
            else:
                # 所有其他类型会在这里进行注册
                super().__setattr__(name, value)

    # setstate、 getstate方法会在序列化的过程中被pickle自己调用
    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __getattr__(self, name: str) -> Union[Parameter, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_modules" in self.__dict__:
            _modules = self.__dict__["_modules"]
            if name in _modules:
                return _modules[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        extra_lines = []
        extra_repr = self.extra_repr()

        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "):" + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("

        if lines:
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"

        return main_str


class Linear(Module):
    """
    线性层: y = x^T*W + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            Tensor.empty((out_features, in_features)), **factory_kwargs
        )

        if bias:
            self.bias = Parameter(Tensor.zeros(out_features), **factory_kwargs)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        x = input @ self.weight.T
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Embedding(Module):

    def __init__(
        self,
        num_embeddings: int,
        embeding_dim: int,
        _weight: Optional[Tensor] = None,
        dtype=float_type,
        device=None,
        padding_idx: Optional[int] = None,
    ) -> None:
        """
        词嵌入：也是一种对离散的变量如单词、字等进行编码的方式，使其变为连续的表达。
                和one-hot的表达方式不同的是使用更少的空间，同时维度之间的信息还有相似度的信息。
        args:
            num_embeddings : 词汇表大小
            embedding_dim  : 嵌入维度
        """
        super(Embedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embeding_dim
        self.padding_idx = padding_idx

        # 将训练好的weight加进来
        if _weight is None:
            self.weight = Parameter(
                Tensor.empty((num_embeddings, embeding_dim), dtype=dtype, device=device)
            )
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [
                num_embeddings,
                embeding_dim,
            ], "'Shape of weight does not match num_embeddings and embedding_dim'"
            self.weight = Parameter(_weight, dtype=dtype, device=device)

    def reset_parameters(self) -> None:
        init.uniform_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self):
        if self.padding_idx is not None:
            with no_grad():
                self.weight[self.padding_idx] = 0

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(self.weight, input)

    @classmethod
    def from_pretrained(cls, embeddings: Tensor, freeze=True, padding_idx=None):
        assert (
            embeddings.ndim == 2
        ), "Embedding parameter is expected to be 2-dimensional."
        rows, cols = embeddings.shape
        embedding = cls(
            num_embeddings=rows,
            embeddings_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
        )
        embedding.weight.requires_grad = not freeze
        return embedding

    def extra_repr(self) -> str:
        s = "{numbeddings}, {embedding_dim}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        return s.format(**self.__dict__)


class Sequential(Module):
    def __init__(self, *args) -> None:
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        size = len(self)
        idx = operator.index(idx)
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.keys(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input) -> Tensor:
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> "Sequential":
        self.add_module(str(len(self)), module)
        return self


class ModuleList(Module):
    pass


class Dropout(Module):

    def __init__(self, p: float = 0.5) -> None:
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class ReLU(Module):
    def __init__(self) -> None:
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input)
