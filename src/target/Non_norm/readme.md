# Non_norm 폴더 설명 문서

본 디렉터리(`Non_norm`)에는 **정규화를 적용하지 않은 상태에서 수행한 전체 탐색 실험**의 결과가 저장되어 있다.  
결과물은 모두 Python 스크립트로부터 생성된 `.txt` 형태의 출력이며, 사용된 평가 지표는 다음과 같다.

## 1. 파일 형식
- `acc_*` : **Accuracy만**을 지표로 사용한 결과 파일  
- `loss_acc_*` : **Loss 및 Accuracy**를 모두 사용한 결과 파일

## 2. 파일 목록 및 설명
---

### `acc_adam_AdaGrad.txt`
- 사용 Optimizer: **Adam**, **AdaGrad**  
- 두 Optimizer를 적용한 모델의 Accuracy 결과가 기록된 파일이다.

### `loss_acc_adam.txt`
- 사용 Optimizer: **Adam**  
- Loss와 Accuracy를 모두 평가하여 기록한 출력 파일이다.
