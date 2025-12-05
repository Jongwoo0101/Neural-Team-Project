import pickle
import json
import numpy as np
import os, sys

sys.path.append(os.pardir) 

def pickle_to_json(pkl_path, json_path):
    # 1) pkl 파일 로드
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # 2) numpy 배열 등 JSON 직렬화 불가능한 타입 변환 함수
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()               # numpy → list
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif hasattr(obj, "__dict__"):       
            return convert(obj.__dict__)      # 클래스 객체 → dict로 변환
        else:
            return obj                        # 기본 타입(str, int 등)은 그대로

    # 3) 변환
    converted = convert(data)

    # 4) JSON 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

    print(f"변환 완료: {json_path}")


# 사용 예시
pickle_to_json("network_Team5.pkl", "network_Team5_best.json")
