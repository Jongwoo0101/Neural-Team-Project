# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from data.mnist_reader import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 학습 데이터를 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

# 드롭아웃 사용 유무와 비율 설정 ========================
use_dropout = True  # 드롭아웃을 쓰지 않을 때는 False
dropout_ratio = 0.2
# ====================================================

def __train(use_dropout_flag, dropout_ratio_val):
    network = MultiLayerNetExtend(
        input_size=784, 
        hidden_size_list=[100, 100, 100, 100, 100], 
        output_size=10, 
        weight_init_std='relu', 
        use_batchnorm=True,  # 배치 정규화는 항상 사용
        use_dropout=use_dropout_flag, # 드롭아웃 사용 여부
        dropout_ration=dropout_ratio_val # 드롭아웃 비율
    )
    optimizer = SGD(lr=learning_rate)
    train_acc_list = []
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        # 학습
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)
    
        if i % iter_per_epoch == 0:
            #정확도 측정
            train_acc = network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)

            print(f"epoch:{epoch_cnt} | use_dropout={use_dropout_flag}, Acc={train_acc:.4f}")
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list


# 그래프 그리기==========
x = np.arange(max_epochs)

# 1. 드롭아웃 미사용 (Batch Norm ON, Dropout OFF)
acc_no_dropout = __train(use_dropout_flag=False, dropout_ratio_val=0.5)

# 2. 드롭아웃 사용 (Batch Norm ON, Dropout ON)
acc_with_dropout = __train(use_dropout_flag=True, dropout_ratio_val=0.2) # 드롭아웃 비율은 0.2 사용
    
# 그래프 그리기==========
plt.title("Training Accuracy (Batch Normalization ON)")
plt.plot(x, acc_with_dropout, label=f'BN ON + Dropout ON (p=0.2)', markevery=2)
plt.plot(x, acc_no_dropout, linestyle = "--", label='BN ON + Dropout OFF', markevery=2)

plt.ylim(0, 1.0)
plt.xlim(0, max_epochs)
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend(loc='lower right')
    
plt.show()