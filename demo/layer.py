import numpy as np



class Layer():

    def __init__(self) -> None:
        pass

    def forward(self, x):
        return x

    def backward(self, dout):
        return dout
    
    def update_param(self, lr = 0.02):
        pass

    def clear_grad(self):
        pass
    
    def check_param(self):
        pass    

class Flatten(Layer):
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        out = x.reshape(N, C * H * W)
        self.cache = x.shape
        return out

    def backward(self, dout):
        shape = self.cache
        dx = dout.reshape(shape)
        return dx
    
    def name(self):
        return "Flatten Layer"

class Linear(Layer):

    '''
        x.shape : bz x input_shape
        w.shape : input_shape x output_shape
        b.shape : 1 x output_shape
    '''
    def __init__(self, input_shape, out_shape) -> None:
        self.input, self.output = input_shape, out_shape
        self.w, self.bias= np.random.normal(loc=0.0, scale=0.01, size=(self.input, self.output)), np.zeros([1, self.output]) 
        self.grad_w, self.grad_bias = None, None
        self.cache = None

    '''
        out.shape : bz x output_shape
    '''
    def forward(self, x):
        assert x.shape != [1, self.input], "check input shape"
        self.cache = [x]
        out = np.matmul(x, self.w) + self.bias
        return out
    
    '''
        calculate gradient
        dout.shape : bz x output_shape
        dx.shape : bz x input_shape
    '''
    def backward(self, dout):
        x = self.cache[0]
        self.grad_bias = np.sum(dout, axis = 0)
        self.grad_w = np.dot(x.T, dout)
        dx = np.dot(dout, self.w.T)
        return dx

    def update_param(self, lr):
        self.w = self.w - lr * self.grad_w
        self.bias = self.bias - lr * self.grad_bias
        # for idx, param in enumerate(self.param):
        #     param -= lr * self.grad[idx]

    def clear_grad(self):
        self.grad_w = np.zeros_like(self.grad_w)
        self.grad_bias = np.zeros_like(self.grad_bias)
        # for grad in self.grad:
        #     grad = np.zeros_like(grad)

    def check_param(self):
        print(self.w, self.bias)

    def name(self):
        return "Linear Layer"
    

class Sigmoid(Layer):

    def __init__(self) -> None:
        self.cache = None
        pass
    
    '''
        x.shape : bz x shape
    '''

    def forward(self, x):
        # x = np.maximum(x, 50)
        np.clip(x, -100, 100)
        out = 1.0 / (1 + np.exp(-x))
        self.cache = [out]
        return out

    def backward(self, dout):
        tmp = self.cache[0]
        return dout * tmp * (1 - tmp)
    
    def name(self):
        return "Sigmoid Layer"

class ReLu(Layer):

    def __init__(self) -> None:
        self.cache = None
        pass
    def forward(self, x):
        self.cache = [x]
        return np.maximum(x, 0)
    def backward(self, dout):
        x = self.cache[0]
        dout[x <= 0] = 0
        return dout


class Softmax(Layer):
    def __init__(self) -> None:
        self.cache = None
        pass

    def forward(self, x):
        input_max = np.max(x, axis = 1, keepdims = True)
        exp_x = np.exp(x - input_max)
        out = exp_x / np.sum(exp_x, axis = 1, keepdims = True)
        self.cache = [x, input_max, out]
        return out

    def backward(self, dout):
        # return dout
        dx = np.zeros_like(dout)
        _, _, out = self.cache
        # batch size loop
        for i in range(dout.shape[0]):
            # feature loop
            for j in range(dout.shape[1]):
                for k in range(dout.shape[1]):
                    if j == k:
                        dx[i][j] += out[i][j] * (1 - out[i][j]) * dout[i][k]
                    else:
                        dx[i][j] += -out[i][j] * out[i][k] * dout[i][k]
        return dx

    def name(self):
        return "Softmax Layer"


class Loss():
    def __init__(self) -> None:
        self.cache = None

    def forward(self, x, label):
        self.cache = [x, label]
        return x
    
    def backward(self):
        return self.cache[0]

class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        self.cache = None

    def forward(self, x, label):
        batch_size = x.shape[0]
        self.cache = [x, label]
        return -np.sum(label * np.log(x + 1e-7)) / batch_size

    def backward(self):
        x, label = self.cache[0], self.cache[1]
        batch_size = x.shape[0]
        # return (x - label) / batch_size    this gradient is softmax with cross entropy
        dx = - label / x
        return dx / batch_size

    def name(self):
        return "Cross Entropy Loss Layer"
    

class SoftmaxLossLayer(Layer):
    def __init__(self):
        print('\tSoftmax loss layer.')
    def forward(self, input):  # 前向传播的计算
        # TODO：softmax 损失层的前向传播，计算输出结果
        input_max = np.max(input, axis=1, keepdims=True)
        input_exp = np.exp(input - input_max)
        self.prob = input_exp / np.sum(input_exp, axis=1, keepdims=True)
        return self.prob
    def get_loss(self, label):   # 计算损失
        self.batch_size = self.prob.shape[0]
        self.label_onehot = np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size), label] = 1.0
        loss = -np.sum(np.log(self.prob) * self.label_onehot) / self.batch_size
        return loss
    def backward(self):  # 反向传播的计算
        # TODO：softmax 损失层的反向传播，计算本层损失
        bottom_diff = (self.prob - self.label_onehot) / self.batch_size
        return bottom_diff