from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear, Gelu
from loss import EuclideanLoss, SoftmaxCrossEntropyLoss, HingeLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import models
import matplotlib.pyplot as plt
import time

train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
# model, loss = models.Model_Linear_Gelu_1_SoftmaxCrossEntropyLoss()

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 40e-3,
    'weight_decay': 1e-5,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 51,
    'disp_freq': 50,
    'test_epoch': 5
}


# added
def plot_func(pos, target_list, title, xlabel, ylabel, name_list):
    ax = plt.subplot(pos)
    plt.sca(ax)
    for target, model_name in zip(target_list, name_list):
        plt.plot(target, label=model_name)
    plt.legend(loc=5, fontsize='xx-small', framealpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


loss_training_list = []
acc_training_list = []
loss_test_list = []
acc_test_list = []
name_list = []
time_list = []
# modified
for name, model, loss in models.model_list:
    start_time = time.time()
    loss_history_training = []
    acc_history_training = []
    loss_history_test = []
    acc_history_test = []
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        # modified
        l, a = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        loss_history_training.extend(l)
        acc_history_training.extend(a)
        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            # modified
            l, a = test_net(model, loss, test_data, test_label, config['batch_size'])
            loss_history_test.append(l)
            acc_history_test.append(a)
    loss_training_list.append(loss_history_training)
    acc_training_list.append(acc_history_training)
    loss_test_list.append(loss_history_test)
    acc_test_list.append(acc_history_test)
    name_list.append(name)
    time_list.append(time.time() - start_time)
plt.figure(figsize=(26, 18))
plot_func(321, loss_training_list, 'Loss on Training Set', 'Every {} Iterations'.format(config['disp_freq']), 'Loss', name_list)
plot_func(322, acc_training_list, 'Accuracy on Training Set', 'Every {} Iterations'.format(config['disp_freq']),
          'Accuracy', name_list)
plot_func(323, loss_test_list, 'Loss on Test Set', 'Epoch', 'Loss', name_list)
plot_func(324, acc_test_list, 'Accuracy on Test Set', 'Epoch', 'Accuracy', name_list)
ax_time = plt.subplot(325)
plt.sca(ax_time)
plt.bar(name_list, time_list, width=0.1)
plt.ylabel('Run Time (s)')
plt.title('Run Time Comparison')
plt.savefig('./plots/HingeLoss2.png')
plt.show()
