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
        '''Your codes here'''
        return (input - target)/input.shape[0]


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END


class HingeLoss(object):
    def __init__(self, name, threshold=0.05):
        self.name = name

    def forward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

    def backward(self, input, target):
        # TODO START
        '''Your codes here'''
        pass
        # TODO END

