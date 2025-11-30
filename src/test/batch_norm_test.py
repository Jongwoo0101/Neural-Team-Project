import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.mnist_reader import load_mnist
from models.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD

# 0. MNIST 데이터 읽기
x_train, t_train = load_mnist('../data', kind='train')
x_test, t_test = load_mnist('../data', kind='t10k')

# 샘플 수 축소
x_train = x_train[:1000]
t_train = t_train[:1000]

# 학습 설정
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01

# 드롭아웃 옵션
default_dropout_ratio = 0.2


# 1. 학습 함수
def train(use_dropout_flag, dropout_ratio_val):
    """BatchNorm 포함한 MLP 학습 후 정확도 리스트 반환"""

    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std='relu',
        use_batchnorm=True,
        use_dropout=use_dropout_flag,
        dropout_ration=dropout_ratio_val
    )

    optimizer = SGD(lr=learning_rate)
    train_acc_list = []

    iter_per_epoch = max(train_size // batch_size, 1)
    epoch_cnt = 0

    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 파라미터 업데이트
        grads = network.gradient(x_batch, t_batch)
        optimizer.update(network.params, grads)

        # 에폭마다 정확도 계산
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)

            print(f"epoch {epoch_cnt:02d} | dropout={use_dropout_flag} | acc={train_acc:.4f}")

            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break

    return train_acc_list


# 2. 두 조건 비교 학습 실행
x = np.arange(max_epochs)

print("\n=== 1) BatchNorm ON + Dropout OFF ===")
acc_no_dropout = train(use_dropout_flag=False, dropout_ratio_val=0.0)

print("\n=== 2) BatchNorm ON + Dropout ON (0.2) ===")
acc_with_dropout = train(use_dropout_flag=True, dropout_ratio_val=default_dropout_ratio)


# 3. 그래프
plt.title("Training Accuracy (BatchNorm ON)")
plt.plot(x, acc_with_dropout, label='BN ON + Dropout ON (p=0.2)', markevery=2)
plt.plot(x, acc_no_dropout, linestyle='--', label='BN ON + Dropout OFF', markevery=2)

plt.ylim(0, 1.0)
plt.xlim(0, max_epochs)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc='lower right')

plt.show()
