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
    """실험 결과를 깔끔하게 정렬된 텍스트 줄로 요약하여 출력한다."""
    # 필드 너비를 지정하여 문자열을 포매팅 (VS Code 고정폭 폰트 전제)
    # 예시:<7 -> 7칸 왼쪽 정렬, :>4 -> 4칸 오른쪽 정렬

    #hyperparameter_tuning.py의 output를 출력하기 위한 파일 경로
    file_path = "../target/ex.txt"
    with open(file_path, 'w', encoding='utf-8') as f:
        # 배치 정규화(BN)/Dropout 플래그를 Yes/No 문자열로 변환 (출력 간결화)
        bn_status = 'T' if params.get('use_batchnorm') else 'F'
        drop_status = 'T' if params['dropout_ration'] > 0 else 'F'
        setting_str = (
            f"Optimizer: {params['optimizer']:<7}, "          # Adam,AdaGrad -> 7칸 확보
            f"LR: {params['lr']:<7}, "                  # 0.1,0.001... -> 7칸 확보
            f"Batch: {params['batch_size']:>4}, "               # 64, 128, 256... -> 4칸 확보 (오른쪽 정렬)
            f"Iters: {params['max_iterations']:>4}, "            # 500, 1000... -> 4칸 확보 (오른쪽 정렬)
            f"Depth: {len(params['hidden_size_list']):>1}, "            # 1~6... -> 1칸 확보
            f"Act/Init: {params['activation']}/{params['weight_init_std']:<5}, "    # relu/relu, sigmoid/sigmoid -> 8칸 확보|relu/relu만 있을땐 5칸
            f"L2: {params['weight_decay_lambda']:<7}, "                    # 1e-08, 0 등 -> 7칸 확보
            f"BN: {bn_status:<1}, "                                      # T/F -> 1칸 확보 
            f"Drop: {drop_status:<1}, "                       # T/F -> 1칸 확보
            f"Drop R: {params['dropout_ration']:<5.1f} "                  # Dropout Ratio (0.0 ~ 0.6) -> 5칸 확보, 소수점 첫째 자리까지
        )
        result_str = f"| FINAL LOSS: {final_loss:.6f} | Test ACC: {final_acc:.4f} |"

        f.write(f"{setting_str} {result_str}\n")