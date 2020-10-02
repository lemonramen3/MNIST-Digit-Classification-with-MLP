from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d


def Model_Linear_Relu_1_EuclideanLoss():
    name = '1_Relu_EuclideanLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = EuclideanLoss(name='loss')
    return name, model, loss


def Model_Linear_Relu_1_SoftmaxCrossEntropyLoss():
    name = '1_Relu_SoftmaxCrossEntropyLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return name, model, loss


def Model_Linear_Relu_1_HingeLoss():
    name = '1_Relu_HingeLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = HingeLoss(name='loss')
    return name, model, loss


def Model_Linear_Sigmoid_1_EuclideanLoss():
    name = '1_Sigmoid_EuclideanLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = EuclideanLoss(name='loss')
    return name, model, loss


def Model_Linear_Sigmoid_1_SoftmaxCrossEntropyLoss():
    name = '1_Sigmoid_SoftmaxCrossEntropyLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return name, model, loss


def Model_Linear_Sigmoid_1_HingeLoss():
    name = '1_Sigmoid_HingeLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = HingeLoss(name='loss')
    return name, model, loss


def Model_Linear_Gelu_1_EuclideanLoss():
    name = '1_Gelu_EuclideanLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = EuclideanLoss(name='loss')
    return name, model, loss


def Model_Linear_Gelu_1_SoftmaxCrossEntropyLoss():
    name = '1_Gelu_SoftmaxCrossEntropyLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return name, model, loss


def Model_Linear_Gelu_1_HingeLoss():
    name = '1_Gelu_HingeLoss'
    model = Network()
    model.add(Linear('fc1', 784, 256, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 256, 10, 0.01))
    loss = HingeLoss(name='loss')
    return name, model, loss


def Model_Linear_Relu_2_EuclideanLoss():
    name = '2_Relu_EuclideanLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Relu('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = EuclideanLoss(name='loss')
    return name, model, loss


def Model_Linear_Sigmoid_2_EuclideanLoss():
    name = '2_Sigmoid_EuclideanLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Sigmoid('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = EuclideanLoss(name='loss')
    return name, model, loss


def Model_Linear_Gelu_2_EuclideanLoss():
    name = '2_Gelu_EuclideanLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Gelu('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = EuclideanLoss(name='loss')
    return name, model, loss


def Model_Linear_Relu_2_SoftmaxCrossEntropyLoss():
    name = '2_Relu_SoftmaxCrossEntropyLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Relu('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return name, model, loss


def Model_Linear_Sigmoid_2_SoftmaxCrossEntropyLoss():
    name = '2_Sigmoid_SoftmaxCrossEntropyLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Sigmoid('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return name, model, loss


def Model_Linear_Gelu_2_SoftmaxCrossEntropyLoss():
    name = '2_Gelu_SoftmaxCrossEntropyLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Gelu('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = SoftmaxCrossEntropyLoss(name='loss')
    return name, model, loss


def Model_Linear_Relu_2_HingeLoss():
    name = '2_Relu_HingeLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Relu('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Relu('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = HingeLoss(name='loss')
    return name, model, loss


def Model_Linear_Sigmoid_2_HingeLoss():
    name = '2_Sigmoid_HingeLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Sigmoid('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Sigmoid('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = HingeLoss(name='loss')
    return name, model, loss


def Model_Linear_Gelu_2_HingeLoss():
    name = '2_Gelu_HingeLoss'
    model = Network()
    model.add(Linear('fc1', 784, 441, 0.01))
    model.add(Gelu('a1'))
    model.add(Linear('fc2', 441, 196, 0.01))
    model.add(Gelu('a2'))
    model.add(Linear('fc3', 196, 10, 0.01))
    loss = HingeLoss(name='loss')
    return name, model, loss


model_list = [Model_Linear_Relu_2_HingeLoss(),
              Model_Linear_Sigmoid_2_HingeLoss(),
              Model_Linear_Gelu_2_HingeLoss()

              ]
