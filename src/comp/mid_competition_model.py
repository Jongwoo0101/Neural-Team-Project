import numpy as np
import pickle
import os, sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.optimizer import Adam
from common.multi_layer_net_extend import MultiLayerNetExtend
from dataset.fashion_mnist import load_fashion_mnist

# 1. 데이터 로드
x_train, t_train = load_fashion_mnist('../dataset', kind='train')
x_test, t_test = load_fashion_mnist('../dataset', kind='t10k')

# flatten 여부 자동 체크
if len(x_train.shape) == 3:  # (N, 28, 28) → flatten
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

input_size = x_train.shape[1]
output_size = 10
train_size = x_train.shape[0]

print("train shape:", x_train.shape)
print("test shape:", x_test.shape)


# 2. 하이퍼파라미터 설정 (최적 조합)
num_hidden_layers = 6
hidden_size = 512

hidden_size_list = [hidden_size] * num_hidden_layers

batch_size = 256
learning_rate = 0.0001
max_epochs = 40

weight_decay_lambda = 0.0001
use_batchnorm = "True"
dropout_ratio = 0.2


# 3. 모델 생성
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


# 4. 학습 설정
iter_per_epoch = max(train_size // batch_size, 1)
max_iter = iter_per_epoch * max_epochs

# Validation split
val_ratio = 0.1
val_size = int(train_size * val_ratio)

x_val = x_train[:val_size]
t_val = t_train[:val_size]

x_train2 = x_train[val_size:]
t_train2 = t_train[val_size:]

train_size = x_train2.shape[0]

print(f"[INFO] Train: {train_size}, Val: {val_size}, Test: {x_test.shape[0]}")
print(f"[INFO] Model: Input({input_size}) -> {hidden_size_list} -> Output({output_size})")
print("---------------------------------------------------------\n")


# 5. Early Stopping & LR Scheduler 변수
best_val_loss = float('inf')
best_params = None
patience = 5
patience_counter = 0

min_lr = 1e-6


def reduce_lr():
    global optimizer
    new_lr = max(optimizer.lr * 0.5, min_lr)
    optimizer.lr = new_lr
    print(f"[Scheduler] Reduce LR → {new_lr:.8f}")


# 6. Gradient clipping
def clip_gradients(grads, clip_value=1.0):
    for key in grads.keys():
        grad = grads[key]
        norm = np.linalg.norm(grad)
        if norm > clip_value:
            grads[key] = grad * (clip_value / norm)
    return grads


# 7. Training Loop
start = time.time()
for i in range(max_iter):
    batch_mask = np.random.choice(train_size, batch_size, replace=False)
    x_batch = x_train2[batch_mask]
    t_batch = t_train2[batch_mask]

    # Backprop
    grads = network.gradient(x_batch, t_batch)
    
    # gradient clipping
    grads = clip_gradients(grads, clip_value=1.0)

    # Parameter update
    optimizer.update(network.params, grads)

    # Epoch 단위 출력
    if (i + 1) % iter_per_epoch == 0:
        epoch = (i + 1) // iter_per_epoch
        train_loss = network.loss(x_batch, t_batch)
        
        # Validation loss
        val_loss = network.loss(x_val, t_val)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = pickle.dumps(network.params)  # deep copy
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
        
        # ReduceLROnPlateau
        if patience_counter == 3:
            reduce_lr()
        
        # Stop
        if patience_counter >= patience:
            print("\n[Early Stopping Triggered]")
            break

end = time.time()
print(f"\nTotal training time: {end - start:.2f} sec")


# 8. Best parameters 복구 후 테스트 정확도 계산
network.params = pickle.loads(best_params)
test_acc = network.accuracy(x_test, t_test)
print(f"\n[Test Accuracy] {test_acc:.4f}")


# 9. 모델 저장
save_path = "network_Team5.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(network, f)

print(f"\nSaved model → {save_path}")