# 자주 사용하는 함수들 모음
# 코드 재사용 방지, 메모리 최적화 시도용

import numpy as np

def sigmoid(x):
    # overflow 방지 버전
    pos_mask = (x >= 0)
    neg_mask = ~pos_mask
    z = np.zeros_like(x, dtype=np.float64)
    z[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    z[neg_mask] = exp_x / (1 + exp_x)
    return z

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

