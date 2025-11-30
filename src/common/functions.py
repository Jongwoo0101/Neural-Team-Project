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

def to_one_hot(t, num_classes=10):
    return np.eye(num_classes)[t]

def summarize_results(params, final_loss, final_acc):
    """실험 결과를 정렬된 텍스트로 간결하게 요약 출력."""

    # 안전한 접근(get) 사용 및 상태 플래그 계산
    use_bn = params.get('use_batchnorm', False)
    dropout_ratio = params.get('dropout_ration', 0.0)
    activation = params.get('activation', 'relu')
    weight_init = params.get('weight_init_std', 'relu')

    bn_status = 'T' if use_bn else 'F'
    drop_status = 'T' if dropout_ratio > 0 else 'F'

    # 출력 필드 구성
    fields = [
        f"Optimizer: {params.get('optimizer', 'N/A'):<7}",
        f"LR: {params.get('lr', 'N/A'):<7}",
        f"Batch: {params.get('batch_size', 0):>4}",
        f"Epochs: {params.get('max_epochs', 0):>3}",
        f"Iters: {params.get('max_iterations', 0):>4}",
        f"Depth: {len(params.get('hidden_size_list', [])):>1}",
        f"Act/Init: {activation}/{weight_init:<5}",
        f"L2: {params.get('weight_decay_lambda', 0):<7}",
        f"BN: {bn_status}",
        f"Drop: {drop_status}",
        f"Drop R: {dropout_ratio:<5.1f}"
    ]

    # 한 줄로 합침
    setting_str = ", ".join(fields)
    result_str = f"| FINAL LOSS: {final_loss:.6f} | Test ACC: {final_acc:.4f} |"

    print(f"{setting_str} {result_str}")
