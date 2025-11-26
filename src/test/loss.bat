@REM output_log.txt에서 loss_acc_log.txt으로 변경
@REM 출력을 리다이렉션할 때 파일이 없다면 자동으로 생성한다.
@REM 출력 경로를 src\target 폴더로 변경시에 ..\target\name.txt으로 변경
python hyperparameter_tuning_loss.py > loss_acc_log.txt