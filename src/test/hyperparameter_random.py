"""
1. accuracy_comparison.py와 달라짐 점
바꿔야하는 하이퍼 파라미터를 자동으로 변경 및 학습(GridSearch)
2. 실행
이 코드는 실행시에 output이 길기에 결과를 txt파일에 표시한다.
이를 위해 src/test>위치에서 loss.bat으로 실행한다.
(cmd창에 출력을 원할 경우 python hyperparameter_tuning.py으로 실행)
loss.bat

3. 출력에 대한 설명
loss_acc_log.txt 출력: 4*2*2*2*6*6=1152줄
Optimizer: Adam
LR: 0.1, 0.01, 0.001, 0.0001
Batch: 128, 256
Iters: 500, 1000
Depth: 1, 2, 3, 4, 5, 6
Act/Init: relu/relu, sigmoid/sigmoid
L2: 0, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0
4. 실행 위치 기반 세팅
cd src
cd test
python hyperparameter_tuning.py
loss.bat

5. 하이퍼파라미터 결과
-h)batch_size: 32, 64, 128, 256에서 64, 128, 256으로 변경
-h)max_iterations: 500, 1000, 4000, 5000에서 500, 1000으로 변경
를 적용한 상태로 처음 돌린 후 ACC와 Loss를 기준으로 내림차순한 결과를 설명한다.

loss_acc_log.txt:accuracy를 기준으로 내림차순한 파일
Opt: Adam으로만
batch_size: 128, 256으로 변경
Depth: 높을수록 좀 더 잘 나오는 경향 있지만 그대로
L2: 굉장히 soft, 그대로

6. 변경사항
-summarize_results에서 Act와 Init를 따로 파일 출력했는데, Act/Init으로 통합했다.
-summarize_results에서 시각적 정렬을 위해 f-string 포매팅을 적용했다.(변경을 대비하기 위해 넉넉하게 잡음)
--ex) 기존엔 LR: 0.01와 LR: 0.0001의 경우 밀리기에 필드 너비를 고정하는 f-string 포매팅 적용했다.
-h)batch_size: 32, 64, 128, 256에서 64, 128, 256으로 변경
-h)max_iterations: 500, 1000, 4000, 5000에서 500, 1000으로 변경
--일반 컴퓨터 환경에서 h)의 두 가지 사항을 이전 코드로 수행시에 2*4*5*5*2*6*6=14,400의 경우의 수이고 max_iterations가 4000,5000이 있기에 loss계산을 안해도 40-240시간이 걸릴 수 있다.
--이를 돌리기 위해선 딥러닝 라이브러리가 GPU 가속을 지원해야하기에 TensorFlow/PyTorch를 사용해야한다. TensorFlow로 테스트시에 common.optimizer의 optimizer을 쓸 경우 GPU를 쓰지 못하였다.
hyperparameter_tuning_loss.py에서 hyperparameter_tuning.py으로 이름을 바꾸었다.
-acc만 사용했던 것과 다르게 loss도 기준을 세워서 뒤에_loss를 붙였지만 이후 과정을 위해 생략한다.
배치 정규화와 Dropout에 대한 코드도 추가
-Dropout에 대한 True,False는 (drop_r > 0)으로 처리
-이에 따른 경우의 수 계산
--epoch 도입의 경우: 3*2*2*1*4*6*2*7=4032
--epoch 도입하지 않은 경우: 3*2*1*1*4*6*2*7=2016
"""
import sys, os, time
from datetime import timedelta
import numpy as np
import pandas as pd   # ← CSV 저장을 위해 추가

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import *
from common.functions import summarize_results
from dataset.fashion_mnist import load_fashion_mnist
from itertools import product

# 시간 측정 시작
start_time = time.time()

# ==================== MNIST 데이터 ======================
x_train, t_train = load_fashion_mnist('../dataset', kind='train')
x_test, t_test = load_fashion_mnist('../dataset', kind='t10k')

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 결과를 빠르게 얻기 위해 훈련 데이터/에폭을 줄임
x_train = x_train[:1000]
t_train = t_train[:1000]



train_size = x_train.shape[0]
if t_test.ndim != 1:
    y_true_test = np.argmax(t_test, axis=1)
else:
    y_true_test = t_test



# Validation split
val_ratio = 0.1
val_size = int(train_size * val_ratio)

x_val = x_train[:val_size]
t_val = t_train[:val_size]

x_train2 = x_train[val_size:]
t_train2 = t_train[val_size:]
train_size = x_train2.shape[0]

x_train = x_train2
t_train = t_train2
# 
NUM_SEARCH_ITERATIONS = 300
# =============== 공통 하이퍼파라미터 ===============
# Random Search 범위 정의
COMMON_HPARAMS_RANGES = {
    'learning_rate_log': (-2.0, -5.0),# 1e-4 ~ 1e-2
    'weight_decay_lambda_log': (-2.699, -6), # 1e-5 ~ 1e-4
    #이산값
    'batch_size': [128],
    'dropout_rations': [0.2],#[0, 0.1, 0.2, 0.3]
    'use_batchnorm': [True],
    'max_epochs': [10], # 에폭 기준을 추가
    'hidden_depths': [5], # Depth를 리스트로 정의
}
# (Sampling이 아닌 실제 값)
MODEL_HPARAMS = {
    'activation_init_sets': [
        {'activation': 'relu', 'weight_init_std': 'relu'}
    ],
    'hidden_size_lists': {
        5: [64, 64, 64, 64, 64],
    }
}

OPTIMIZER_TO_USE = Adam
OPTIMIZER_NAME = 'Adam'

# CSV 저장용 결과 리스트
results = []
# ==============================================================
# 3. 무작위 조합 순회 (Random Search Loop)
# ==============================================================
for search_iter in range(NUM_SEARCH_ITERATIONS):
    # 3-1. 하이퍼파라미터 무작위 샘플링
    
    # 로그 스케일 값 샘플링 후 10의 지수로 변환
    log_lr = np.random.uniform(*COMMON_HPARAMS_RANGES['learning_rate_log'])
    lr = 10**log_lr
    # lr=0.0048204543415823 #good: 0.007991645
    log_l2 = np.random.uniform(*COMMON_HPARAMS_RANGES['weight_decay_lambda_log'])
    l2 = 10**log_l2
    # l2=0.0000104211334480552
    # 이산 값 리스트에서 무작위 선택
    bs = np.random.choice(COMMON_HPARAMS_RANGES['batch_size'])
    max_e = np.random.choice(COMMON_HPARAMS_RANGES['max_epochs'])
    drop_r = np.random.choice(COMMON_HPARAMS_RANGES['dropout_rations'])

    # 고정 값 선택
    depth = COMMON_HPARAMS_RANGES['hidden_depths'][0]
    use_bn = COMMON_HPARAMS_RANGES['use_batchnorm'][0]
    act_init = MODEL_HPARAMS['activation_init_sets'][0]
    
    # 3-2. 훈련에 필요한 반복 횟수 계산 (Epoch 기반)
    iter_per_epoch = max(train_size // bs, 1)
    max_i = max_e * iter_per_epoch 
    
    # 3-3. 현재 실험 파라미터 딕셔너리 생성
    current_params = {
        'optimizer': OPTIMIZER_NAME,
        'lr': lr,
        'batch_size': int(bs),
        'max_iterations': int(max_i),
        'max_epochs': int(max_e),
        'hidden_size_list': MODEL_HPARAMS['hidden_size_lists'][depth],
        'activation': act_init['activation'],
        'weight_init_std': act_init['weight_init_std'],
        'weight_decay_lambda': l2,
        'use_batchnorm': use_bn,
        'dropout_ration': drop_r
    }

    train_loss_list = []
    try:
        # 모델 초기화
        network = MultiLayerNetExtend(
            input_size=784,
            hidden_size_list=current_params['hidden_size_list'],
            output_size=10,
            activation=current_params['activation'],
            weight_init_std=current_params['weight_init_std'],
            weight_decay_lambda=current_params['weight_decay_lambda'],
            use_batchnorm=current_params['use_batchnorm'],
            use_dropout=(drop_r > 0),
            dropout_ration=current_params['dropout_ration']
        )
        optimizer = OPTIMIZER_TO_USE(lr=current_params['lr'])

        # 훈련 반복
        for i in range(current_params['max_iterations']):
            batch_mask = np.random.choice(train_size, current_params['batch_size'])
            x_batch = x_train[batch_mask]# x_train2 대신 x_train 사용
            t_batch = t_train[batch_mask]# t_train2 대신 t_train 사용

            grads = network.gradient(x_batch, t_batch)
            optimizer.update(network.params, grads)

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

        # 평가
        final_loss = train_loss_list[-1] if train_loss_list else np.inf
        network.use_dropout = False
        # final_acc = network.accuracy(x_test, t_test)
        # 검증데이터, 하이퍼파라미터 성능 평가를 x_val, t_val로 변경
        final_val_acc = network.accuracy(x_val, t_val)

        summarize_results(current_params, final_loss, final_val_acc)

        # CSV 저장용 데이터 추가
        results.append({
            **current_params,
            'final_loss': float(final_loss),
            'final_val_acc': float(final_val_acc) # 검증 정확도 저장
        })
    except Exception as e:
        params_info = f"LR:{lr:.2e}, Batch:{bs}, Epoch:{max_e}, Depth:{depth}, L2:{l2:.2e}"
        print(f"|ERROR| Failed Experiment ({params_info}) | Exception: {e}")
        
        results.append({
            **current_params,
            'final_loss': np.inf,
            'final_acc': 0.0
        })

# =============== CSV 저장 (추가된 부분) ===============
df = pd.DataFrame(results)
csv_path = "hyperparam_results.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

print(f"\nCSV 저장 완료 → {csv_path}\n")

# =============== 총 소요 시간 출력 ===============
end_time = time.time()
elapsed_seconds = end_time - start_time
td = timedelta(seconds=elapsed_seconds)
time_str = str(td)
if '.' in time_str:
    time_str = time_str[:time_str.find('.') + 4]
else:
    time_str += '.000'

print("============================================================")
print(f"Total Experiment Combinations: {len(results)}")
print(f"Total Elapsed Time: {time_str}")
print("============================================================")
