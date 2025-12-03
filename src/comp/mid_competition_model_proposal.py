import numpy as np
import os, sys,pickle

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.optimizer import Adam
from models.multi_layer_net_extend import MultiLayerNetExtend
from data.mnist_reader import load_mnist
from common.util import shuffle_dataset
import matplotlib.pyplot as plt
def print_hyperparameters():
    print("==========================================")
    print(" Model & Training Hyperparameters ")
    print("==========================================")
    print(f"**신경망 구조:** 층={num_hidden_layers}개 (노드 {hidden_size}개), 출력={output_size}개")
    print(f"**활성화 함수:** ReLU (He 초기화 사용)")
    print(f"**최적화 함수:** Adam (lr={learning_rate})")
    print("------------------------------------------")
    print(f"**훈련 에폭 수:** {max_epochs}")
    print(f"**배치 크기:** {batch_size}")
    print(f"**L2 규제 (Weight Decay):** {weight_decay_lambda}")
    print(f"**배치 정규화 (BN) 사용:** {use_batchnorm}")
    print(f"**드롭아웃 비율:** {dropout_ratio}")
    print("------------------------------------------")
    
    # 검증 데이터 사용 여부 출력
    if va_rate > 0:
        print(f"**데이터 분할:** 훈련:{100-int(va_rate*100)}%, 검증:{int(va_rate*100)}% (검증 사용)")
        print(f"  └ 훈련 데이터 크기: {x_train.shape[0]}개")
        print(f"  └ 검증 데이터 크기: {x_val.shape[0]}개")
    else:
        print(f"**데이터 분할:** 검증 데이터 **사용 안 함**")
        print(f"  └ 전체 훈련 데이터 크기: {x_train_all.shape[0]}개")
    print("==========================================")




# 2. 데이터 로드 (여기서는 MNIST 예시)
x_train_all, t_train_all = load_mnist('../data', kind='train')
x_test, t_test = load_mnist('../data', kind='t10k')

x_train_all = x_train_all.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

input_size = x_train_all.shape[1]
output_size = 10

x_train_all,t_train_all=shuffle_dataset(x_train_all,t_train_all)
va_rate=0.0
va_num=int(x_train_all.shape[0]*va_rate)

x_val=x_train_all[:va_num];t_val=t_train_all[:va_num]
x_train=x_train_all[va_num:];t_train=t_train_all[va_num:]
train_acc_list=[];va_acc_list=[];test_acc_list=[]
# 3. 신경망/하이퍼파라미터 설정
num_hidden_layers = 6
hidden_size = 256
hidden_size_list = [hidden_size] * num_hidden_layers

batch_size = 256
learning_rate = 0.01#0.001
max_epochs = 10#100

weight_decay_lambda = 1e-6#1e-8 - 1e-6
use_batchnorm = True
dropout_ratio = 0.1 # good(0.1-0.2)
print_hyperparameters() # 함수 호출
# 4. 모델 생성
network = MultiLayerNetExtend(
    input_size=input_size,
    hidden_size_list=hidden_size_list,
    output_size=output_size,
    activation='relu',
    weight_init_std='he',
    weight_decay_lambda=weight_decay_lambda,
    use_batchnorm=use_batchnorm,
    dropout_ration=dropout_ratio
)

optimizer = Adam(lr=learning_rate)

train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)
max_iter = iter_per_epoch * max_epochs

# 5. 학습 루프
for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 역전파
    grads = network.gradient(x_batch, t_batch)

    # Adam 업데이트
    optimizer.update(network.params, grads)

    # epoch마다 출력
    if (i + 1) % iter_per_epoch == 0:
        epoch = (i + 1) // iter_per_epoch
        # if epoch == 50:
        #     optimizer.lr *= 0.1  # 0.01 -> 0.001
        #     print(f"--- [LR Decay] Epoch {epoch}: Learning Rate decreased to {optimizer.lr:.6f} ---")
        loss = network.loss(x_batch, t_batch)
        train_acc = network.accuracy(x_train, t_train)
        # val_acc=network.accuracy(x_val,t_val)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        # va_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        # print(f"[Epoch {epoch}] loss: {loss:.4f}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}")
        print(f"[Epoch {epoch}] loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}, LR: {optimizer.lr:.4f}")

# 그래프 그리기==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
# plt.plot(x, va_acc_list, marker='d', label='train', markevery=10)

plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0.75, 1.0)
plt.title("Model Accuracy Comparison")
plt.legend(loc='lower right')
plt.grid(True)

# 3. Adam 내부 LR 그래프 (Iter vs. lr_t)
plt.subplot(3, 1, 3) 
iter_data = optimizer.iter_history
lr_t_data = optimizer.lr_t_history

# 전체 이터레이션 수가 너무 많으면 그래프를 샘플링하여 그리는 것이 좋습니다.
# 여기서는 모든 이터레이션을 플롯합니다.
plt.plot(iter_data, lr_t_data, linewidth=0.5, color='blue', label='Adaptive Learning Rate (lr_t)')

plt.xlabel("Iteration (Batch Step)")
plt.ylabel("Effective Learning Rate ($\mathbf{lr}_t$)")
# y축을 로그 스케일로 설정하여 초반의 급격한 변화를 더 잘 보이게 합니다.
plt.yscale('log')
plt.title("Adam Optimizer's Effective Learning Rate ($\mathbf{lr}_t$) per Iteration")
plt.legend()
plt.grid(True, which="both", linestyle="--")


plt.show()

# # 6. 학습된 파라미터 pkl 파일로 저장
# save_path = "model_params.pkl"
# with open(save_path, 'wb') as f:
#     pickle.dump(network.params, f)

# print(f"\n✅ 학습된 모델 파라미터가 '{save_path}' 로 저장되었습니다.")


# 7. (선택) 저장된 pkl 파일 불러오는 예시
# 새 모델 생성 후
'''
new_network = MultiLayerNetExtend(
    input_size=input_size,
    hidden_size_list=hidden_size_list,
    output_size=output_size,
    activation='relu',
    weight_init_std='he'
)

# 불러오기
with open("model_params.pkl", "rb") as f:
    new_network.params = pickle.load(f)

print("모델 파라미터 로드 완료.")


'''

