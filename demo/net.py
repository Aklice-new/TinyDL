import numpy as np
from typing import Any, Sequence


from layer import Loss, Layer

class Net():

    def __init__(self, layers : Sequence, loss : Loss) -> None:
        self.layers = layers
        self.loss_layer = loss
    
    def step(self, lr:float = 0.02):
        for layer in self.layers:
            layer.update_param(lr)

    def backward(self):
        dout = self.loss_layer.backward()
        for layer in self.layers[::-1]:
            dout = layer.backward(dout)

    def clear_grad(self):
        for layer in self.layers:
            layer.clear_grad()

    def __call__(self, x, label) -> Any:
        for layer in self.layers:
            x = layer.forward(x)

        loss = self.loss_layer.forward(x, label)
        return loss   

    def test(self, x) -> Any:
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x, axis=1)

    def check_param(self):
        for layer in self.layers:
            layer.check_param()