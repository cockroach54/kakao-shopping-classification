# shopping-classification

python 3.6, tensorflow 기준으로 작성되었습니다.

## used library
 - tqdm 4.19.8
 - numpy 1.14.2
 - pandas 0.20.3
 - tensorflow 1.12.0

## data
 - data_org 폴더에 원본 데이터 저장
 - tmp 폴더에 셔플된 데이터 청크 및 char-cnn용 vocab저장

## preprocess
 - 데이터 셔플링 및 validation용 데이터 선출

## training
    Model
    - char-cnn
        - Yoon Kim
        - char은 출현 빈도순으로 2998개 + <UNK> + <PAD>  총 3000개 사용
        - [product] + [brand] + [model] + [maker] 합쳐서 150자 짜름
    - bi-directional
        - 위와 반대로 뒤부터 시작
    - one by one cnn
    - img feature
        - attention with char cnn max pooled features
    - price
        - 결측치 보정 위해 (1,0)으로 구분되는 컬럼 추가
        - log10 취해서 사용 
    - architecture

    - 190epoch 학습
## inference