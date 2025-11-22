"""
multi_layer_net_test.py와 달라짐 점
: 두 그래프의 범위 지정, 정확도 계산 그래프 추가
cd src
cd test
python accuracy_comparison.py
"""
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.util import smooth_curve
from models.multi_layer_net import MultiLayerNet
from common.optimizer import *
from data.mnist_reader import load_mnist


# 0. MNIST 데이터 읽기==========
x_train, t_train = load_mnist('../data', kind='train')
x_test, t_test = load_mnist('../data', kind='t10k')

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
iter_per_epoch = max(train_size / batch_size, 1)

# 1. 실험용 설정==========
optimizers = {}
# optimizers['SGD'] = SGD(lr=0.01)
# optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam(lr=0.001)
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
test_acc = {} # 테스트 정확도 목록을 저장
for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []
    test_acc[key] = []
# 2. 훈련 시작==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    # 2-1. 옵티마이저별 학습: 기울기 계산>가중치 업데이트>미니배치에 대한 손실 계산>해당 옵티마이저의 train_loss 리스트에 추가
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    # 2-2. 에폭 단위로 정확도 계산 (모든 optimizers에 대해 수행)
    if i % iter_per_epoch == 0:
        print( "===========" + "epoch:" + str(int(i / iter_per_epoch)) + "===========")
        # 새로운 루프를 추가하여 모든 옵티마이저의 정확도를 계산
        for key in optimizers.keys():
            # 테스트 정확도 계산 및 기록
            acc = networks[key].accuracy(x_test, t_test)
            test_acc[key].append(acc)
            current_loss = networks[key].loss(x_batch, t_batch)
            # 손실 값 출력
            # print(f"{key}: loss={current_loss:.4f}, test acc={acc:.4f}")
    #2-3. iteration이 100증가마다 optimizers의 loss출력
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

# 2.5. 학습 후 최종 정확도 계산 및 출력==========
for key in optimizers.keys():
    network = networks[key]
    test_acc_ch = network.accuracy(x_test, t_test)
    print(f"[{key}] 테스트 정확도: {test_acc_ch:.4f}")

# 3. 그래프 그리기==========
# ax1에는 첫 번째 그래프 (손실), ax2에는 두 번째 그래프 (정확도)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
markers = {"AdaGrad": "s", "Adam": "D"}
# 3-1.첫 번째 그래프 optimizer에 대한 loss-----
min_x1, max_x1 = 0, max_iterations
# 3-1-1.Y축 하한선 설정: 모든 손실 값의 최소값보다 0.05 낮게 (최종 수렴 지점 확대), 모든 옵티마이저의 평활화된 손실 데이터 수집
smoothed_losses = []
for key in optimizers.keys():
    # smooth_curve 함수를 사용하여 평활화된 데이터를 리스트에 추가
    smoothed_losses.extend(smooth_curve(train_loss[key]))
# 3-1-2. 평활화된 데이터의 최소값으로 min_y1 설정
min_y1 = max(0.0, np.min(smoothed_losses) - 0.05)
max_y1 = 1.0
# 손실을 기록한 리스트 (train_loss[key])는 매 반복마다 계산된 원래의 손실 값, smooth_curve 함수를 적용한 것과는 차이가 있어 위 방식을 채택
# all_losses = [loss for key in optimizers.keys() for loss in train_loss[key]]
# min_y1 = max(0.0, np.min(all_losses) - 0.05)
# max_y1=1.0

x = np.arange(max_iterations)
for key in optimizers.keys():
    ax1.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
ax1.set_xlabel("iterations")
ax1.set_ylabel("loss")
ax1.set_xlim(min_x1, max_x1)
ax1.set_ylim(min_y1, max_y1)
ax1.set_title("Training Loss")
ax1.grid(True)
ax1.legend()
# 3-2.두 번째 그래프 optimizer에 대한 정확도-----
epochs = np.arange(len(test_acc['AdaGrad']))
min_x2, max_x2=0, len(epochs) - 1
# 모든 옵티마이저의 정확도 리스트에서 가장 낮은 값
all_accuracies = [acc for key in optimizers.keys() for acc in test_acc[key]]
min_y2, max_y2 = min(np.min(all_accuracies) - 0.05, 1) , 1.0
min_y2 = max(0.0, min_y2)

for key in optimizers.keys():
    ax2.plot(epochs, test_acc[key], marker=markers[key], markevery=3, label=key)
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
ax2.set_xlim(min_x2, max_x2)
ax2.set_ylim(min_y2, max_y2)
ax2.set_title("Optimizer Comparison: Test Accuracy")
ax2.legend()
ax2.grid(True)

fig.tight_layout() # 서브플롯 간의 간격 조정
plt.show()#두 그래프 show