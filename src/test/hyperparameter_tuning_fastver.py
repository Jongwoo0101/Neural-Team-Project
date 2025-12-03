'''
다양한 모델 구조 및 학습 조건이 MNIST 분류 성능에 미치는 영향을 분석하고,
최적의 하이퍼파라미터 조합을 탐색한다
'''
import sys, os, time, traceback
from datetime import timedelta
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import Adam
from common.functions import summarize_results
from dataset.fashion_mnist import load_fashion_mnist
from itertools import product

# MNIST Load
x_train, t_train = load_fashion_mnist('../data', kind='train')
x_test, t_test = load_fashion_mnist('../data', kind='t10k')

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

train_size = x_train.shape[0]

# test subset (속도)
x_test_small = x_test[:1000]
t_test_small = t_test[:1000]

# Hyperparameters
COMMON_HPARAMS = {
    'learning_rate': [1e-3, 1e-4],
    'batch_size': [128, 256],
    'max_iterations': [200],
}

MODEL_HPARAMS = {
    'hidden_size_lists': {
        3: [64, 64, 64],
        5: [64, 64, 64, 64, 64],
    },
    'weight_decay_lambda': [0, 1e-4],
    'use_batchnorm': [False],
    'dropout_rations': [0, 0.3],   # 기존 코드에 맞춰 'dropout_ration' 사용 중이라면 그대로 유지
}

ACT_INIT = {'activation': 'relu', 'weight_init_std': 'relu'}
OPTIMIZER = Adam
OPT_NAME = "Adam"

results = []

# Product combination
all_combinations = product(
    COMMON_HPARAMS['learning_rate'],
    COMMON_HPARAMS['batch_size'],
    COMMON_HPARAMS['max_iterations'],
    MODEL_HPARAMS['hidden_size_lists'].keys(),
    MODEL_HPARAMS['weight_decay_lambda'],
    MODEL_HPARAMS['use_batchnorm'],
    MODEL_HPARAMS['dropout_rations'],
)

start_time = time.time()

# Train Loop
for lr, bs, max_it, depth, l2, use_bn, drop_r in all_combinations:
    params = {
        'optimizer': OPT_NAME,
        'lr': lr,
        'batch_size': bs,
        'max_iterations': max_it,
        'max_epochs': max_it,   # <-- 핵심 수정: 기존 코드가 이 키를 필요로 함
        'hidden_size_list': MODEL_HPARAMS['hidden_size_lists'][depth],
        'activation': ACT_INIT['activation'],
        'weight_init_std': ACT_INIT['weight_init_std'],
        'weight_decay_lambda': l2,
        'use_batchnorm': use_bn,
        'dropout_ration': drop_r   # 기존 코드와 동일한 키 이름 유지
    }

    try:
        # 모델 초기화
        net = MultiLayerNetExtend(
            input_size=784,
            hidden_size_list=params['hidden_size_list'],
            output_size=10,
            activation=params['activation'],
            weight_init_std=params['weight_init_std'],
            weight_decay_lambda=params['weight_decay_lambda'],
            use_batchnorm=params['use_batchnorm'],
            use_dropout=(drop_r > 0),
            dropout_ration=params['dropout_ration']
        )

        optimizer = OPTIMIZER(lr=params['lr'])
        train_loss_list = []

        # Train iter
        idx = 0
        for i in range(max_it):

            # O(1) batch slicing (shuffle 제거 — 속도 우선)
            batch_idx = np.arange(idx, idx + bs) % train_size
            idx += bs

            x_batch = x_train[batch_idx]
            t_batch = t_train[batch_idx]

            grads = net.gradient(x_batch, t_batch)
            optimizer.update(net.params, grads)

            # loss는 20 iter 단위로만 계산
            if i % 20 == 0:
                loss = net.loss(x_batch, t_batch)
                train_loss_list.append(loss)

        # Evaluation
        net.use_dropout = False
        final_loss = train_loss_list[-1] if train_loss_list else float('inf')
        final_acc = net.accuracy(x_test_small, t_test_small)

        # summarize_results에 params 전체를 넘김 (함수 내부가 특정 키를 기대할 가능성 있음)
        summarize_results(params, final_loss, final_acc)

        results.append({**params, 'final_loss': float(final_loss), 'final_acc': float(final_acc)})

    except Exception as e:
        # 전체 traceback 출력하여 원인 파악을 쉽게 함
        print(f"[ERROR] params={params} | Exception: {e}")
        traceback.print_exc()
        results.append({**params, 'final_loss': float('inf'), 'final_acc': 0})

# Save CSV
df = pd.DataFrame(results)
out_csv = "hyperparam_results_fast_fix.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nCSV 저장: {out_csv}\n")

# Time
elapsed = timedelta(seconds=(time.time() - start_time))
print("====================================================")
print(f"Total combinations: {len(results)}")
print(f"Elapsed time: {elapsed}")
print("====================================================")
