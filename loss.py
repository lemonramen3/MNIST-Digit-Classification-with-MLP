from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        # print(input)
        # print(target)
        # print(np.linalg.norm(input - target, axis=1))
        # print(input - target)
        # print(input.shape, target.shape)
        # print(input.shape, target.shape)
        norm = np.linalg.norm(input - target, axis=1)  # (100,10)
        # print(norm.shape[0])
        return np.sum(norm * norm)/(norm.shape[0] * 2.)
        # TODO END

    def backward(self, input, target):
        # TODO STSRT
        '''Your codes here'''
        return (input - target) / input.shape[0]
        # TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        # input: (batch_size, 10) target: (batch_size, 10)
        h = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)  # (batch_size, 10)
        E = -np.sum(target * np.log(h), axis=1, keepdims=True)  # (100, 1)
        return np.sum(E) / input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        # Reference: https://blog.csdn.net/weixin_43846347/article/details/94363273
        '''Your codes here'''
        return (input - target) / input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, threshold=0.05):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        delta = 5.
        x_t = np.max(np.where(target == 1, input, 0.), axis=1, keepdims=True)
        a = np.array(np.maximum(0., delta - x_t + input))
        h = np.where(input == 1, 0., a)
        self.saved_tensor = h
        return np.sum(h) / input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        h = self.saved_tensor
        filter= np.where(h > 0, 1, 0)
        # Counter for max(0, delta-x_tn+x_k)>0
        counter = np.sum(filter, axis=1, keepdims=True)
        grad_x_tn = np.zeros(h.shape) - counter
        return np.where(target == 1, grad_x_tn, filter)/input.shape[0]
        # TODO END

