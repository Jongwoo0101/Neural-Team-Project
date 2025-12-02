import numpy as np
import pickle
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.optimizer import Adam
from models.multi_layer_net_extend import MultiLayerNetExtend
from data.mnist_reader import load_mnist

# 2. 데이터 로드 및 정규화
x_train, t_train = load_mnist('../data', kind='train')
x_test, t_test = load_mnist('../data', kind='t10k')

# 데이터 정규화 (0-1 범위로)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

input_size = x_train.shape[1]
output_size = 10

# 3. 신경망/하이퍼파라미터 설정
num_hidden_layers = 5  # 6층 이하
hidden_size = 512  # 256 → 512로 증가
hidden_size_list = [hidden_size] * num_hidden_layers

batch_size = 128  # 256 → 128 (더 자주 업데이트)
learning_rate = 0.0005  # 0.001 → 0.0005 (안정적인 학습)
max_epochs = 30  # 20 → 30 (충분한 학습)

weight_decay_lambda = 0.0001  # 0.0005 → 0.0001 (정규화 완화)
use_batchnorm = True
dropout_ratio = 0.2  # 0.01 → 0.2 (적절한 정규화)

# 4. 모델 생성
network = MultiLayerNetExtend(
    input_size=input_size,
    hidden_size_list=hidden_size_list,
    output_size=output_size,
    activation='relu',
    weight_init_std='he',
    weight_decay_lambda=weight_decay_lambda,
    use_batchnorm=use_batchnorm,
    use_dropout=True,
    dropout_ration=dropout_ratio
)

optimizer = Adam(lr=learning_rate)

train_size = x_train.shape[0]
iter_per_epoch = max(train_size // batch_size, 1)
max_iter = iter_per_epoch * max_epochs

print(f"총 반복 횟수: {max_iter}, 에포크당 반복: {iter_per_epoch}")
print(f"네트워크 구조: 입력({input_size}) -> {hidden_size_list} -> 출력({output_size})")
print(f"학습 시작...\n")

# 5. 학습 루프
best_test_acc = 0.0
for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 역전파
    grads = network.gradient(x_batch, t_batch)
    
    # Adam 업데이트
    optimizer.update(network.params, grads)
    
    # epoch마다 출력
    if (i + 1) % iter_per_epoch == 0:
        epoch = (i + 1) // iter_per_epoch
        loss = network.loss(x_batch, t_batch, train_flg=True)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        
        print(f"[Epoch {epoch:2d}/{max_epochs}] loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")
        
        # 최고 정확도 갱신
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # 최고 성능 모델 저장
            with open("best_model_params.pkl", 'wb') as f:
                pickle.dump(network.params, f)

print(f"\n학습 완료!")
print(f"최고 테스트 정확도: {best_test_acc:.4f}")

# 6. 최종 학습된 파라미터 저장
save_path = "model_params.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(network.params, f)
print(f"최종 모델 파라미터가 '{save_path}' 로 저장되었습니다.")

# 최종 평가
final_train_acc = network.accuracy(x_train, t_train)
final_test_acc = network.accuracy(x_test, t_test)
print(f"\n최종 훈련 정확도: {final_train_acc:.4f}")
print(f"최종 테스트 정확도: {final_test_acc:.4f}")