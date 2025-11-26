@REM output_log.txt에서 ..\target\sgd.txt으로 변경
@REM 출력을 리다이렉션할 때 파일이 없다면 자동으로 생성한다.
@REM 1. 실행할 스크립트(.py)의 경로: 상위 폴더(src)에서 test 폴더로 이동하여 실행
@REM 2. 출력 파일(.txt)의 경로: 상위 폴더(src)에서 target 폴더로 이동하여 저장
..\test\hyperparameter_tuning_loss.py > ..\target\loss_acc_log.txt
python hyperparameter_tuning_loss.py > ..\target\sgd.txt