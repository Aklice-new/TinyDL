import math
from collections import defaultdict, Counter
from typing import List, Dict

from tinydl import no_grad
from tinydl.tensor import Tensor


# 优化器 Solver，用于将梯度更新到tensor，不同的子类，实现了不同的更新方法：SGD,Adam,AdaGrad等
class Optimizer:

    def __init__(self, params, defaults) -> None:
        """
        args:
            params : 待优化的参数列表，Tensor或者dict {'params':nn.Parameter, 'lr':lr:float}
            defaults : 包含优化器默认值的字典
        """
        self.defaults = defaults
        # 参数分组
        self.params_groups = []
        self.state = defaultdict(dict)

        param_groups = list(params)

        # 如果不是字典
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        return {
            "defaults": self.defaults,
            "state": self.state,
            "param_groups": self.params_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string

    def zero_grad(self) -> None:
        for group in self.params_groups:
            for param in group["params"]:
                param.zero_grad()

    def step(self) -> None:
        raise NotImplementedError

    def add_param_group(self, param_group: dict):
        assert isinstance(param_group, dict), "param group must be a dict"
        params = param_group["params"]

        if isinstance(params, Tensor):
            param_group["params"] = [params]
        else:
            param_group["params"] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        self.params_groups.append(param_group)


# 学习率调整器：用于在训练过程中，对学习率进行动态的调整，不同的子类实现了不同的调整方法
class LRScheduler:

    def __init__(
        self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False
    ):
        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.params_groups:
                # 设置初始值
                group.setdefault("initial_lr", group["lr"])
        else:
            for i, group in enumerate(optimizer.params_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        f"param 'initial_lr' is not specified "
                        "in param_groups[{i}] when resuming an optimizer"
                    )

        # 保存各个参数的初始学习率
        self.base_lrs = [group["lr"] for group in optimizer.params_groups]
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._initial_step()

    def _initial_step(self):
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step()

    def get_lr(self):
        raise NotImplementedError

    def get_last_lr(self):
        return self._last_lr

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """如果 is_verbose为True， 打印当前的学习率"""
        if is_verbose:
            if epoch is None:
                print(f"Adjusting learning rate of group {group} to {lr:.4e}.")
            else:
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    f"Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.4e}."
                )

    def step(self, epoch=None):
        self._step_count += 1
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        for i, data in enumerate(zip(self.optimizer.params_groups, self.get_lr())):
            param_group, lr = data
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr, epoch)

        # 保存这次学习率
        self._last_lr = [group["lr"] for group in self.optimizer.params_groups]



class SGD(Optimizer):
    '''
    SGD 随机梯度下降
    '''
    def __init__(self, params, lr:float=1e-3, weight_decay=0) -> None:
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self) ->None:
        with no_grad():
            for group in self.params_groups:
                weight_decay = group['weight_decay']
                lr = group['lr']

                for param in group['params']:
                    d_p = param.grad
                    if weight_decay != 0:
                        d_p += weight_decay * param
                    param.add_(d_p, alpha=-lr)
                

class SGDMomentum(Optimizer):
    '''
    带有动量的SGD
    '''
    def __init__(self, params, lr:float=1e-3, weight_decay=0, beta=0.9)->None:
        defaults = dict(lr=lr, weight_decay=weight_decay, beta=beta)
        super().__init__(params, defaults)

        for group in self.params_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum'] = Tensor.zeros_like(p)
                state['step'] = Tensor(0.)
    
    def _init_step(self, group, params_with_grad, grads, state_momentums, state_steps):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                state_momentums.append(state['momentum'])
                state_steps.append(state['step'])

    def step(self):
        for group in self.params_groups:
            params_with_grad = []
            grads = []
            state_momentums = []
            state_steps = []

            self._init_step(group, params_with_grad, grads, state_momentums, state_steps)
            
            weight_decay = group['weight_decay']
            lr = group['lr']
            beta = group['beta']

            for (param, grad, momentum, step_t) in zip(params_with_grad, grads, state_momentums, state_steps):
                step_t += 1
                
                if weight_decay!=0:
                    grad = grad + weight_decay * param.grad
                # m_t = \beta * m_{t - 1} + grad
                momentum.mul_(beta).add_(grad, alpha=1)
                # x_{t} = x_{t - 1} - lr * m_{t}
                param.add_(momentum, alpha=-lr)


class Adagrad(Optimizer):
    '''
        Adagrad是为了改进固定的学习率
        对于更新不频繁的参数，我们希望更新的步长大些，学习率高些，反之亦然。
    '''

    def __init__(self, params, lr=1e-2, lr_decay=0, initial_accumulater_value = 0, eps=1e-10, weight_decay=0) -> None:
        defaults = dict(
            lr = lr, 
            eps = eps,
            weight_decay= weight_decay, 
            lr_decay = lr_decay,
            initial_accumulater_value = initial_accumulater_value
        )
        super().__init__(params, defaults)

        for group in self.params_groups:
            for param in group['params']:
                state = self.state[param]
                state['step'] = Tensor(0.)
                init_value = initial_accumulater_value
                state['sum'] = Tensor.full_like(param, init_value)
    
    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                
                state = self.state[p]

                state_sums.append(state['sum'])
                state_steps.apend(state["step"])

    def step(self) -> None:
        with no_grad():
            for group in self.params_groups:
                params_with_grad = []
                grads = []
                state_sums = []
                state_steps = []

                self._init_group(group, params_with_grad, grads, state_sums, state_steps)

                weight_decay = group['weight_decay']
                lr = group['lr']
                lr_decay = group['lr_decay']
                eps = group['eps']

                for (param, grad, state_sum, step_t) in zip(params_with_grad, grads, state_sums, state_steps):
                    # 更新step
                    step_t += 1
                    step = step_t.item()
                    if weight_decay != 0:
                        grad = grad + weight_decay * param.grad
                    
                    clr = lr / (1 + (step  - 1) * lr_decay)

                    state_sum.addcmul_(grad, grad, value=1)
                    std = state_sum.sqrt().add_(eps)
                    param.addcdiv_(grad, std, value = -clr)


class Adam(Optimizer):
    '''
    Adam是为了解决在Adagrad的过程中，分母不断变大导致后面参数更新很慢的问题
    '''

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) -> None:
        defaults = dict(
            lr = lr,
            betas = betas,
            eps = eps, 
            weight_decay = weight_decay
        )
        super().__init__(params, defaults)

        for group in self.params_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = Tensor(0.)
                state['exp_avg'] = Tensor.zeros_like(p)
                state['exp_avg_sq'] = Tensor.zeros_like(p)
    
    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                exp_avgs.append(state['exp_avgs'])
                exp_avg_sqs.append(state['exp_avg_sqs'])
                state_steps.append(state['step'])
    
    def step(self) ->None:
        for group in self.params_groups:
            
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            weight_decay = group['weight_decay']
            lr = group['lr']
            eps = group['eps']

            self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps)

            for (param, grad, exp_avg, exp_avg_sq, state_step) in zip(params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps):

                state_step += 1
                if weight_decay != 0:
                    grad = grad + weight_decay * param.grad
                
                # m_t = beta1 * m_{t - 1} + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                # v_t = beta2 * v_{t - 1} + (1 - beta2) * grad^2
                exp_avg_sq.mul(beta2).addcmul_(grad, grad, value=1 - beta2)

                step = step.item()
                
                # 修正 m_t, v_t的系数
                # \hat{m_t} = \frac{m_t}{1 - beta1^step}
                # \hat{v_t} = \frac{v_t}{1 - beta2^step}
                bias_correctoin1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correctoin1
                bias_sqrt = math.sqrt(bias_correction2)

                denom = (exp_avg_sq.sqrt() / bias_sqrt).add_(eps)

                # x_t = x_{t - 1} - lr * \frac{\hat{m_{t - 1}}}{\sqrt(\hat{v_{t - 1}} + eps)}
                param.addcdiv_(exp_avg, denom, value = -step_size)


class ExponentialLR(LRScheduler):

    def __init__(self, optimizer: Optimizer, gamma,  last_epoch: int = -1, verbose: bool = False):
        """
        每个epoch通过gamma衰减每个parameter group的学习率，当last_epoch=-1，学习率设为初始值
        :param optimizer: 优化器
        :param gamma: 学习率衰减的乘法因子
        :param last_epoch: 最后一次epoch的索引
        :param verbose: 是否为每次更新打印信息
        """
        super().__init__(optimizer, last_epoch, verbose)
        self.gamma = gamma
    
    def get_lr(self):
        if self.last_epoch == 0:
            # 第一轮学习率即为初始学习率
            return [group['lr'] for group in self.optimizer.params_groups]
        # 学习率开始衰减
        return [group['lr'] * self.gamma for group in self.optimizer.params_groups]

class StepLR(LRScheduler):
    '''
    固定step_size 个epoch才进行一次衰减
    '''
    def __init__(self, optimizer: Optimizer,step_size,gamma=0.1, last_epoch: int = -1, verbose: bool = False):
        super().__init__(optimizer, last_epoch, verbose)
        self.gamma = gamma
        self.step_size = step_size
    
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
             # 第一轮学习率即为初始学习率，非整除的epoch学习率不衰减
            return [group['lr'] for group in self.optimizer.params_groups] 
        # 学习率epoch整除的轮数衰减
        return [group['lr'] * self.gamma for group in self.optimizer.params_groups]