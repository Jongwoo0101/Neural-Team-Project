# Log 폴더 설명 문서

본 디렉터리(`log`)는 `target` 폴더에 저장된 실험 결과(.txt 파일)를 기반으로 수행한 **후처리 분석 결과**를 포함한다.  
각 파일은 Sorting 또는 통계 기반 분석을 위해 생성되었다.

## 1. 파일 목록 및 설명
---

### `sort_acc.txt`
- 지표: **Accuracy**  
- Accuracy 값을 기준으로 내림차순 정렬한 결과를 포함한다.

### `sort_loss.txt`
- 지표: **Loss**  
- Loss 기준 내림차순 정렬 결과를 정리한 파일이다.

### `sort_percentiles.txt`
- 구성 요소: `sort_acc.txt`, `sort_loss.txt`  
- 상위 **Top {1%, 3%, 6%, 12%}** 모델에 대한 종합적인 분포 및 통계 분석을 포함하는 파일이다.
