import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor

class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        return np.maximum(0, input)

        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        grad_output[self._saved_tensor <= 0] = 0
        return grad_output
        # TODO END

class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        # TODO START
        # Reference: https://blog.csdn.net/qq_33200967/article/details/79759284
        '''Your codes here'''
        self._saved_for_backward(1./(1. + np.exp(-input)))
        return self._saved_tensor
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        return grad_output * self._saved_tensor * (1 - self._saved_tensor)
        # TODO END

class Gelu(Layer):
    def __init__(self, name):
        super(Gelu, self).__init__(name)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        temp = np.sqrt(2/np.pi) * (input + 0.044715 * input ** 3)
        tanh = np.tanh(temp)
        return 0.5 * input * (1 + tanh)
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        input = self._saved_tensor
        tmp = 0.0356774 * input ** 3 + 0.797885 * input
        p1 = 0.5 * np.tanh(tmp)
        sech = 2 / (np.exp(tmp) + np.exp(-tmp))
        p2 = (0.0535161 * input ** 3 + 0.398942 * input) * sech ** 2
        # slope = (self.forward(input + 1e-5) -self.forward(input))/1e-5
        # print(p1+p2+0.5 - slope)
        return grad_output * (p1 + p2 + 0.5)
        # TODO END

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        '''Your codes here'''
        self._saved_for_backward(input)
        # print(input.shape)
        return input.dot(self.W) + self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        '''Your codes here'''
        self.grad_W = self._saved_tensor.T.dot(grad_output)  # (in_num, out_num)
        self.grad_b = grad_output.sum(axis=0)
        # print(grad_output.shape)# (100,10)
        return grad_output.dot(self.W.T)
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W
        # print(lr* self.diff_W)
        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
