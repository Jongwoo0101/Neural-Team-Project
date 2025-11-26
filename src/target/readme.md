# Target 폴더 설명 문서

본 디렉터리(`target`)에는 **정규화(Normalization)를 적용한 전체 탐색 실험**의 결과물이 저장되어 있다.  
각 파일은 Python 기반 실험 스크립트의 출력으로 생성된 `.txt` 형식의 결과이며, 모델의 성능을 평가하기 위한 지표로 **Loss**와 **Accuracy**를 사용하였다.

## 1. 파일 형식
- 파일명 규칙: `loss_acc_*`
- 주요 지표:  
  - **Loss**  
  - **Accuracy**

## 2. 파일 목록 및 설명
---

### `loss_acc_momentum.txt`
- 사용 Optimizer: **Momentum**  
- 정규화를 적용한 모델의 학습 결과(Loss 및 Accuracy)를 기술한 출력 파일이다.

### `loss_acc_adam.txt`
- 사용 Optimizer: **Adam**  
- Adam Optimizer 기반 모델의 학습 결과가 기록되어 있다.

### `loss_acc_sgd.txt`
- 사용 Optimizer: **SGD(Stochastic Gradient Descent)**  
- SGD를 적용한 경우의 Loss 및 Accuracy 변화를 포함한다.
