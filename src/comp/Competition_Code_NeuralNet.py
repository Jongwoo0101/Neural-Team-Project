
#
# 여러분은,
# 다음과 같이 test_data.pkl 파일을 불러와서
#


import os, sys
import numpy as np
from PIL import Image
import pickle

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

from common.util import *
from dataset.fashion_mnist import load_fashion_mnist



# test_data.pkl 파일 불러오기
with open('test_data.pkl', 'rb') as f:
    data = pickle.load(f)

x_test = data["x_test"]
t_test = data["t_test"]

print("x_test:", x_test.shape)
print("t_test:", t_test.shape)

# 자기 팀의 network 파일 불러오기
with open('network_Team5.pkl', 'rb') as f: # 5조에서 제출한 network 파일을 사용한 예시
    network = pickle.load(f)

# 자기 팀의 accuray 구하여 출력하기
accuracy = network.accuracy(x_test, t_test)
print(f"accuracy:{accuracy:.4f}", )



    # with open('network_Team1.pkl', 'wb') as f:
    #    pickle.dump(trainer.network, f)


# 문의 사항이 있으면, 
# 즉시 경천관 404호(031-280-3661)로 오거나 저에게 전화해서 물어보시기 바랍니다.

# 수업시간에 최종발표를 모두 마치고, 
# 팀별로 저에게 노트북을 가지고 와서, accuracy 출력 값을 제게 보여 주고 
# 팀 network.pkl 파일과 Competition_Code.py 파일을 게시판에 제출해 주시기 바랍니다.

# 화이팅!
