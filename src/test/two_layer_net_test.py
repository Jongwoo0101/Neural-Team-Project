import sys,os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dataset.fashion_mnist import load_fashion_mnist
from common.functions import to_one_hot
from models.two_layer_net import TwoLayerNet

x_train, t_train = load_fashion_mnist('../data', kind='train')

x_test, t_test = load_fashion_mnist('../data', kind='t10k')

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = to_one_hot(t_train[batch_mask])
    
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"train acc: {train_acc}\ntest acc: {test_acc}")
        
plt.figure(figsize=(10, 4))
plt.plot(np.arange(len(train_loss_list)), train_loss_list, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

epochs = np.arange(len(train_acc_list))
plt.figure(figsize=(10, 4))
plt.plot(epochs, train_acc_list, label='Train Accuracy')
plt.plot(epochs, test_acc_list, label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()