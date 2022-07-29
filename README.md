# 5G IoT - OD_Framework Docs 2.0

본 문서는 5G IoT 과제를 지원하기 위한 객체 검출 모델 학습 프레임워크로

Poc 별 객체 검출 모델을 학습하기 위한 데이터 수집 및 학습 기능,

Jetson board와 같은 임베디드 보드에서 동작하기 위한 TensorRT 기반 모델 변환,

그리고 변환된 TensorRT engine을 활용한 영상 객체 검출 기능을 지원함으로써 

과제 수행을 위한 Poc별 모델 학습에 편의성을 제공한다. 

(현재 프레임워크 실행 방법에 대한 문서 작업만 진행되었으며, 모듈별 구체적인 설명이 기재된 문서는 추후 작성하여 업로드 예정)

## 시작하기

[프레임워크 셋팅 및 동작 ](Documentation/framework_setting.md)

## 데이터 수집 모듈

[OID_tools](Documentation/OID_tools.md)

[Cityscapes 데이터 지원 ](Documentation/Cityscape_module.md)

## 모델 학습 및 테스트 관련 모듈

[utils_package ](Documentation/utils_package.md)

[Config_directory ](Documentation/Config_directory.md)

## Jetson board 동작을 위한 Detection Package

[Detection_tools_package ](Documentation/Detection_tools_package.md)

[trt_detection](Documentation/trt_detection.md)
