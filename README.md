# Reproducibility-Project
<br>
Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation

<br>

## 프로젝트 소개
- 해당 Repository는 reproducibility임.
- 저장된 사용자 프로필을 사용할 수 없는 상황의 추천 시스템의 패러다임을 제공함.

<br>

## 팀원 구성

<div align="center">
●HunKang
Seoul National University of Science & Technology (SeoulTech) 232 Gongreungno
hunkang10@ds.seoultech.ac.kr

●Syngyeon Tag
Seoul National University of Science & Technology (SeoulTech) 232 Gongreungno
sytak@ds.seoultech.ac.kr
 
●Changsun Lee
Seoul National University of Science & Technology (SeoulTech) 232 Gongreungno
24510056@seoultech.ac.kr

</div>

<br>

## 1. 개발 환경

- 프로그램 사용 : Pytorch, vscode
<br>

## 2. 재현성 요약
-DHCN의 논문에서 제시된 실험 결과 재현하고자 함. \
&nbsp; +원 논문에서 다루지 않은 데이터 세트의 결과를 통합하여 원 논문의 결과를 확장하고자 함.\
&nbsp; +하이퍼파라미터 튜닝에 대한 원 논문의 결과는 특정데이터셋에 국한되어 있었음으로, 하이퍼파라미텅 튜닝을 통한 다양한 데이터 셋으로 확장.\
&nbsp; +"세션 내의 항목은 엄격하게 순차적으로 의존하지 않는다"는 DHCN의 기본 가정과 각 구성 요소의 효과를 검증.
<br>

## 3. 결과 요약
-하나의 데이터 셋을 제외한 나머지 데이터 셋에 대하여 논문과 1% 이내의 정밀도를 재현.\
-새로 적용된 데이터 셋에 대해서도 기준 모델에 비해 더 높은 성능 달성.\
-하이퍼파라미터와 제거 시험에서도 논문과 유사한 결과를 얻음.\
-직접 수정하고 실험한 모델에 대해선 기존의 DHCN보다 더 높은 성능을 달성.\
**=광범위한 실험을 통해 논문의 내용을 확인하고 결과로 검증할 수 있었음.**
