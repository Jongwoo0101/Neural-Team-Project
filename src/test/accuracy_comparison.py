"""
Adam과 AdaGrad비교 코드
multi_layer_net_test.py와 달라진 점
- 두 그래프(손실/정확도)의 범위 지정
- 테스트 정확도 계산 그래프 추가

사용 방법:
cd src
cd test
python accuracy_comparison.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.util import smooth_curve
from models.multi_layer_net import MultiLayerNet
from common.optimizer import *
from dataset.fashion_mnist import load_fashion_mnist

# 0. MNIST 데이터 읽기
x_train, t_train = load_fashion_mnist('../data', kind='train')
x_test, t_test = load_fashion_mnist('../data', kind='t10k')

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
iter_per_epoch = max(train_size // batch_size, 1)

# 1. 실험 설정
optimizers = {
    "AdaGrad": AdaGrad(),
    "Adam": Adam(lr=0.001)
}

networks = {}
train_loss = {}
test_acc = {}

for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],
        output_size=10
    )
    train_loss[key] = []
    test_acc[key] = []

# 2. 훈련
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 옵티마이저별 학습
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    # 에폭마다 테스트 정확도 계산
    if i % iter_per_epoch == 0:
        epoch = int(i / iter_per_epoch)
        print(f"===== epoch: {epoch} =====")

        for key in optimizers.keys():
            acc = networks[key].accuracy(x_test, t_test)
            test_acc[key].append(acc)
            print(f"{key}: loss={loss:.4f}, test acc={acc:.4f}")

    # iteration 100마다 손실 출력
    if i % 100 == 0:
        print(f"===== iteration {i} =====")
        for key in optimizers.keys():
            print(f"{key}: {networks[key].loss(x_batch, t_batch)}")

# 2.5. 학습 후 최종 정확도 출력
for key in optimizers.keys():
    acc = networks[key].accuracy(x_test, t_test)
    print(f"[{key}] 최종 테스트 정확도: {acc:.4f}")

# 3. 그래프 그리기 (Loss + Accuracy)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
markers = {"AdaGrad": "s", "Adam": "D"}

# Loss 그래프
min_x1, max_x1 = 0, max_iterations

# 모든 평활화된 손실 모아서 y축 범위 계산
smoothed_losses = []
for key in optimizers.keys():
    smoothed_losses.extend(smooth_curve(train_loss[key]))

min_y1 = max(0.0, np.min(smoothed_losses) - 0.05)
max_y1 = 1.0

x = np.arange(max_iterations)
for key in optimizers.keys():
    ax1.plot(x, smooth_curve(train_loss[key]),
             marker=markers[key], markevery=100, label=key)

ax1.set_xlabel("Iterations")
ax1.set_ylabel("Loss")
ax1.set_xlim(min_x1, max_x1)
ax1.set_ylim(min_y1, max_y1)
ax1.set_title("Training Loss")
ax1.grid(True)
ax1.legend()

# Accuracy 그래프
epochs = np.arange(len(test_acc['AdaGrad']))
min_x2, max_x2 = 0, len(epochs) - 1

# 모든 정확도에서 최소값 계산
all_accuracies = [acc for key in optimizers.keys() for acc in test_acc[key]]
min_y2 = max(0.0, np.min(all_accuracies) - 0.05)
max_y2 = 1.0

for key in optimizers.keys():
    ax2.plot(epochs, test_acc[key],
             marker=markers[key], markevery=3, label=key)

ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.set_xlim(min_x2, max_x2)
ax2.set_ylim(min_y2, max_y2)
ax2.set_title("Optimizer Comparison: Test Accuracy")
ax2.grid(True)
ax2.legend()

fig.tight_layout()
plt.show()
