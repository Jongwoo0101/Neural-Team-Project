"""
1. hyperparameter_tuning_loss.py와 달라짐 점

--이를 돌리기 위해선 딥러닝 라이브러리가 GPU 가속을 지원해야하기에 TensorFlow/PyTorch를 사용해야한다. TensorFlow로 테스트시에 common.optimizer의 optimizer을 쓸 경우 GPU를 쓰지 못하였다.
2. 실행
이 코드는 실행시에 output이 길기에 결과를 txt파일에 표시한다.
이를 위해 src/test>위치에서 momentum.bat으로 실행한다.
momentum.bat

3. 출력에 대한 설명
loss_acc_log.txt 출력: 4*2*2*2*6*6=1152줄
Optimizer: Momentum
LR: 0.1, 0.01, 0.001, 0.0001
Batch: 128, 256
Iters: 500, 1000
Depth: 1, 2, 3, 4, 5, 6
Act/Init: relu/relu, sigmoid/sigmoid
L2: 0, 1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0
4. 실행 위치 기반 세팅
cd src
cd test
momentum.bat

5. 하이퍼파라미터 결과
-h)batch_size: 32, 64, 128, 256에서 64, 128, 256으로 변경
-h)max_iterations: 500, 1000, 4000, 5000에서 500, 1000으로 변경
를 적용한 상태로 처음 돌린 후 ACC와 Loss를 기준으로 내림차순한 결과를 설명한다.

loss_acc_log.txt:accuracy를 기준으로 내림차순한 파일
Opt: Adam으로만
batch_size: 128, 256으로 변경
Depth: 높을수록 좀 더 잘 나오는 경향 있지만 그대로
L2: 굉장히 soft, 그대로
6. 실행시 뜨는 출력: RuntimeWarning 관련
(base) C:\github_Desktop\w\src\test>python hyperparameter_tuning_loss_momentum.py  1>loss_acc_log_momentum.txt
C:\github_Desktop\w\src\test\..\common\layers.py:59: RuntimeWarning: overflow encountered in dot
  out = np.dot(self.x, self.W) + self.b
C:\github_Desktop\w\src\test\..\common\layers.py:59: RuntimeWarning: invalid value encountered in dot
  out = np.dot(self.x, self.W) + self.b
C:\github_Desktop\w\src\test\..\models\multi_layer_net.py:93: RuntimeWarning: overflow encountered in square
  weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
C:\github_Desktop\w\src\test\..\common\layers.py:65: RuntimeWarning: invalid value encountered in dot
  self.dW = np.dot(self.x.T, dout)
"""
import sys, os, time
from datetime import timedelta#시간 차이를 포맷하기 위해
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.multi_layer_net import MultiLayerNet
from common.optimizer import *
from data.mnist_reader import load_mnist
def summarize_results(params, final_loss, final_acc):
    """실험 결과를 깔끔하게 정렬된 텍스트 줄로 요약하여 출력한다."""
    # 필드 너비를 지정하여 문자열을 포매팅 (VS Code 고정폭 폰트 전제)
    # 예시:<7 -> 7칸 왼쪽 정렬, :>4 -> 4칸 오른쪽 정렬
    setting_str = (
        f"Optimizer: {params['optimizer']:<7}, "          # Adam,AdaGrad -> 7칸 확보
        f"LR: {params['lr']:<7}, "                  # 0.1,0.001... -> 7칸 확보
        f"Batch: {params['batch_size']:>4}, "               # 64, 128, 256... -> 4칸 확보 (오른쪽 정렬)
        f"Iters: {params['max_iterations']:>4}, "            # 500, 1000... -> 4칸 확보 (오른쪽 정렬)
        f"Depth: {len(params['hidden_size_list']):>1}, "            # 1~6... -> 1칸 확보
        f"Act/Init: {params['activation']}/{params['weight_init_std']:<16}, "    # relu/relu, sigmoid/sigmoid -> 16칸 확보
        f"L2: {params['weight_decay_lambda']:<7}"                    # 1e-08, 0 등 -> 7칸 확보
    )
    result_str = f"| FINAL LOSS: {final_loss:.6f} | Test ACC: {final_acc:.4f} |"

    print(f"{setting_str} {result_str}")

start_time = time.time()
# 0. MNIST 데이터 읽기 및 실험 설정 후보 값 정의==========
# 0-1. 데이터 읽기
x_train, t_train = load_mnist('../data', kind='train')
x_test, t_test = load_mnist('../data', kind='t10k')
# 정규화(Momentum을 위한, 성능 개선)
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
train_size = x_train.shape[0]
if t_test.ndim != 1: 
    y_true_test = np.argmax(t_test, axis=1)
else:
    y_true_test = t_test
# 0-2. 공통 하이퍼파라미터
COMMON_HPARAMS = {
    'learning_rate': [1e-1, 1e-2, 1e-3, 1e-4],# 일반적으로 가장 중요.
    'batch_size': [128, 256],# 훈련 안정성과 속도에 영향
    'max_iterations': [500, 1000],# 충분한 수렴 시간 보장 위함
}
# 0-3. MultiLayerNet 모델 설정 후보
MODEL_HPARAMS = {
    # 활성화 함수와 이에 맞는 가중치 초기화 세트> 같이 간다.
    'activation_init_sets': [
        {'activation': 'relu', 'weight_init_std': 'relu'},       # 권장: He 초기값
        {'activation': 'sigmoid', 'weight_init_std': 'sigmoid'}  # 권장: Xavier 초기값
    ],
    # 은닉층 구조: 층의 개수(깊이, index)만 변경 (뉴런 100개 고정)
    'hidden_size_lists': {
        1: [100],
        2: [100, 100],
        3: [100, 100, 100],
        4: [100, 100, 100, 100],
        5: [100, 100, 100, 100, 100],
        6: [100, 100, 100, 100, 100, 100],
    },
    # L2 규제 강도: 0은 규제x
    'weight_decay_lambda': [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4],
}
#0-4. 옵티마이저 목록 (lr은 COMMON_HPARAMS에서 가져오고, 내부 파라미터는 고정)
OPTIMIZER_TO_USE = Momentum # Adam이라는 클래스 자체를 저장하는 변수
OPTIMIZER_NAME = 'Momentum'
# 0-5. 모든 조합을 생성 (product 함수 사용)
# product는 중첩 루프를 대신하여 모든 가능한 조합을 튜플로 반환한다.
# 모델 HPARAMS
activation_inits = MODEL_HPARAMS['activation_init_sets']
hidden_depths = MODEL_HPARAMS['hidden_size_lists'].keys()
l2_lambdas = MODEL_HPARAMS['weight_decay_lambda']

# COMMON HPARAMS
lrs = COMMON_HPARAMS['learning_rate']
batch_sizes = COMMON_HPARAMS['batch_size']
max_iters = COMMON_HPARAMS['max_iterations']

# 6중 루프 시작()
for lr in lrs:
    for bs in batch_sizes:
        for max_i in max_iters:
            for act_init in activation_inits:
                for depth in hidden_depths:
                    for l2 in l2_lambdas:
                        # 현재 실험 파라미터 딕셔너리 생성
                        current_params = {
                            'optimizer': OPTIMIZER_NAME,
                            'lr': lr,
                            'batch_size': bs,
                            'max_iterations': max_i,
                            'hidden_size_list': MODEL_HPARAMS['hidden_size_lists'][depth],
                            'activation': act_init['activation'],
                            'weight_init_std': act_init['weight_init_std'],
                            'weight_decay_lambda': l2
                        }

                        # 2. 단일 조합 훈련 및 평가===========
                        # 훈련 손실 기록 리스트 초기화
                        train_loss_list = []
                        try:
                            # 2.1. 네트워크 및 옵티마이저 초기화
                            network = MultiLayerNet(
                                input_size=784,
                                hidden_size_list=current_params['hidden_size_list'],
                                output_size=10,
                                activation=current_params['activation'],
                                weight_init_std=current_params['weight_init_std'],
                                weight_decay_lambda=current_params['weight_decay_lambda']
                            )
                            # Adam/AdaGrad 옵티마이저 객체 생성
                            optimizer = OPTIMIZER_TO_USE(lr=current_params['lr'])

                            # 2.2. 훈련 루프
                            for i in range(current_params['max_iterations']):
                                batch_mask = np.random.choice(train_size, current_params['batch_size'])
                                x_batch = x_train[batch_mask]
                                t_batch = t_train[batch_mask]
                                
                                grads = network.gradient(x_batch, t_batch)
                                optimizer.update(network.params, grads)
                                
                                # 손실 값 기록 (추가된 부분)
                                loss = network.loss(x_batch, t_batch)
                                train_loss_list.append(loss)
                            # 2.3. 최종 성능 평가
                            # 1. 최종 손실: 기록된 손실 리스트의 마지막 값 사용(리스트가 비어있으면 훈련 실패로 간주.)
                            if train_loss_list:
                                final_loss = train_loss_list[-1] 
                            else:
                                final_loss = np.inf # 무한대로 설정하여 낮은 순위로 밀어냄

                            # 2. 테스트 정확도 (보조 지표로 유지)
                            final_acc = network.accuracy(x_test, t_test)
                            
                            # 2.4. 결과 요약 출력
                            summarize_results(current_params, final_loss, final_acc)
                        except RuntimeWarning:
                            # 실패한 경우, 손실을 무한대로 설정
                            print(f"|FAILED EXPERIMENT| {current_params} | FINAL LOSS: {np.inf:.6f} | Error: Overflow/NaN")
                        
                        except Exception as e:
                            # 기타 예외 처리
                            print(f"|ERROR| {current_params} | FINAL LOSS: {np.inf:.6f} | Exception: {e}")
# 3. 시간 측정
end_time = time.time()
elapsed_seconds = end_time - start_time
# 시간, 분, 초로 변환
td = timedelta(seconds=elapsed_seconds)
# timedelta 객체를 시:분:초.소수점 형식으로 변환하여 출력(td는 기본적으로 HH:MM:SS.microseconds 형식으로 포매팅됨)
time_str = str(td)
if '.' in time_str:
    #round(float,n-1)와 같은 기능,  0:00:01.123456 -> 0:00:01.123)
    time_str = time_str[:time_str.find('.') + 4]
else:
    time_str += '.000' # 소수점이 없는 경우 추가

print(f"\n============================================================")
print(f" Total Experiment Combinations (number of cases): {len(lrs) * len(batch_sizes) * len(max_iters) * len(activation_inits) * len(hidden_depths) * len(l2_lambdas)}")
print(f"⏱ Total Elapsed Time (H:M:S.ms): {time_str}")
print(f"============================================================\n")
