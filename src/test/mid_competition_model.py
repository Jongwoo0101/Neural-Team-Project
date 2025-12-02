import numpy as np
import pickle
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.optimizer import Adam
from models.multi_layer_net_extend import MultiLayerNetExtend
from data.mnist_reader import load_mnist


# 2. 데이터 로드 (여기서는 MNIST 예시)
x_train, t_train = load_mnist('../data', kind='train')
x_test, t_test = load_mnist('../data', kind='t10k')

input_size = x_train.shape[1]
output_size = 10

# 3. 신경망/하이퍼파라미터 설정
num_hidden_layers = 6
hidden_size = 256
hidden_size_list = [hidden_size] * num_hidden_layers

batch_size = 256
learning_rate = 0.001
max_epochs = 20

weight_decay_lambda = 0.0005
use_batchnorm = True
dropout_ratio = 0.1 # 과적합 방지

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
        loss = network.loss(x_batch, t_batch)
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print(f"[Epoch {epoch}] loss: {loss:.4f}, train_acc: {train_acc:.4f}, test_acc: {test_acc:.4f}")

# 6. 학습된 파라미터 pkl 파일로 저장
save_path = "model_params.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(network.params, f)

print(f"\n학습된 모델 파라미터가 '{save_path}' 로 저장되었습니다.")


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

