import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from data.mnist_reader import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *
(x_train,t_train),(x_test, t_test)=load_mnist(normalize=True)
x_train, t_train = load_mnist('../data/fashion/', kind='train')
x_test, t_test =load_mnist('../data/fashion/', kind='t10k')
train_size=x_train.shape[0]
batch_size=128
max_iterations=2000
optimizers={}
optimizers["SGD"]=SGD()
optimizers["Momentum"]=Momentum()
optimizers["AdaGrad"]=AdaGrad()
optimizers["Adam"]=Adam()
net={}
train_loss={}
for key in optimizers.keys():
    net[key]=MultiLayerNet(input_size=784,hidden_size_list=[100,100,100],output_size=10)
    train_loss[key]=[]
for i in range(max_iterations):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]
    for key in optimizers.keys():
        grads=net[key].gradient(x_batch,t_batch)
        optimizers[key].update(net[key].params,grads)
        loss=net[key].loss(x_batch,t_batch)
        train_loss[key].append(loss)
    if i%100==0:
        print(f"===========iteration: {i}===========")
        for key in optimizers.keys():
            loss=net[key].loss(x_batch,t_batch)
            print(f"{key}: {loss}")
markers={"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x=np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x,smooth_curve(train_loss[key]),marker=markers[key],markevery=100,label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0,1)
plt.legend()
plt.show()