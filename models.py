from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


def Model_Linear_Relu_1_EuclideanLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Relu('a2'))
    loss = EuclideanLoss(name='loss')
    return model, loss


def Model_Linear_Relu_1_SoftmaxCrossEntropyLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Relu('a2'))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return model, loss


def Model_Linear_Relu_1_HingeLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Relu('a2'))
    loss = HingeLoss(name='loss')
    return model, loss


def Model_Linear_Sigmoid_1_EuclideanLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Sigmoid('a2'))
    loss = EuclideanLoss(name='loss')
    return model, loss


def Model_Linear_Sigmoid_1_SoftmaxCrossEntropyLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Sigmoid('a2'))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return model, loss


def Model_Linear_Sigmoid_1_HingeLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Sigmoid('a2'))
    loss = HingeLoss(name='loss')
    return model, loss


def Model_Linear_Gelu_1_EuclideanLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Gelu('a2'))
    loss = EuclideanLoss(name='loss')
    return model, loss


def Model_Linear_Gelu_1_SoftmaxCrossEntropyLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Gelu('a2'))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return model, loss


def Model_Linear_Gelu_1_HingeLoss():
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    model.add(Gelu('a2'))
    loss = HingeLoss(name='loss')
    return model, loss


