@REM @REM output_log.txt에서 loss_acc_log.txt으로 변경
@REM @REM 출력을 리다이렉션할 때 파일이 없다면 자동으로 생성한다.
@REM @REM 1. 실행할 스크립트(.py)의 경로: 상위 폴더(src)에서 test 폴더로 이동하여 실행
@REM @REM 2. 출력 파일(.txt)의 경로: 상위 폴더(src)에서 target 폴더로 이동하여 저장
@REM ..\test\hyperparameter_tuning.py > ..\target\loss_acc_log.txt
@REM output_log.txt에서 loss_acc_log.txt으로 변경
@REM 리다이렉션할 때 파일이 없다면 자동으로 생성한다.
@REM 절대 경로를 사용하여 실행 위치와 관계없이 항상 작동하도록 보장

@REM Python 실행 파일의 절대 경로를 지정합니다.
@REM VS Code에서 터미널을 열었을 때 'python'이 환경 변수로 잡혀있다면 생략 가능
@REM C:\Users\tee01\miniconda3\envs\base\python.exe 라고 가정합니다.
set PYTHON_EXE=python

@REM 프로젝트의 루트 경로를 변수로 설정합니다.
set PROJECT_ROOT=C:\github_Desktop\W

@REM 실행 및 출력
%PYTHON_EXE% "%PROJECT_ROOT%\src\test\hyperparameter_tuning.py" > "%PROJECT_ROOT%\target\loss_acc_log.txt"