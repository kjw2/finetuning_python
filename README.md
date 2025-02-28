# 한국어 LLM 파인튜닝 도구

한국어 LLM(Large Language Model) 모델의 파인튜닝을 위한 파이썬 도구입니다. 다양한 한국어 데이터셋을 사용하여 LLM 모델을 fine-tuning할 수 있습니다.

## 목차
1. [설치 방법](#설치-방법)
2. [프로젝트 구조](#프로젝트-구조)
3. [사용 방법](#사용-방법)
4. [설정 파일 가이드](#설정-파일-가이드)
5. [커맨드라인 인자](#커맨드라인-인자)
6. [학습 과정](#학습-과정)
7. [문제 해결](#문제-해결)
8. [FAQ](#faq)

## 설치 방법

### 필수 요구사항
- Python 3.8 이상
- CUDA 지원 GPU (최소 16GB VRAM 권장)
- 충분한 저장 공간 (최소 50GB 권장)

### 1. 저장소 클론
```bash
git clone [repository_url]
cd finetuning_python
```

### 2. 가상환경 생성 (선택사항)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 프로젝트 구조
```
finetuning_python/
├── config.yaml          # 기본 설정 파일
├── finetuning_app.py   # 메인 애플리케이션
├── config_manager.py    # 설정 관리
├── model_manager.py     # 모델 관리
├── data_manager.py      # 데이터 관리
├── training_manager.py  # 학습 프로세스 관리
├── memory_utils.py      # 메모리 최적화 유틸리티
├── exceptions.py        # 커스텀 예외 처리
├── requirements.txt     # 의존성 패키지
└── README.md           # 문서
```

## 사용 방법

### 1. 기본 실행
가장 기본적인 실행 방법입니다. config.yaml의 기본 설정을 사용합니다.
```bash
python finetuning_app.py
```

### 2. 커스텀 설정 파일 사용
자신만의 설정 파일을 사용할 수 있습니다.
```bash
python finetuning_app.py --config path/to/your/config.yaml
```

### 3. 커맨드 라인에서 파라미터 설정
설정 파일 없이 직접 파라미터를 지정할 수 있습니다.
```bash
python finetuning_app.py \
  --model_name "bert-base-uncased" \
  --dataset "imdb" \
  --max_length 1024 \
  --batch_size 16 \
  --epochs 3
```

## 설정 파일 가이드

### config.yaml 기본 구조
```yaml
model:
  name: "bert-base-uncased"
  task: "text-classification"
  output_dir: "./models/fine_tuned_model"
  cache_dir: "./models/pretrained"

data:
  dataset_name: "imdb"
  dataset_cache_dir: "./datasets"
  text_column: "text"
  label_column: "label"
  max_length: 1024
  batch_size: 16

training:
  epochs: 3
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 0
  max_grad_norm: 1.0
  cuda_cleanup_freq: 100

memory:
  gradient_accumulation_steps: 4
  max_gpu_memory_usage: 0.95
  empty_cache_freq: 50
  mixed_precision: true
  optimize_memory_usage: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(levelname)s - %(message)s"
```

### 설정 파라미터 상세 설명

#### 1. 모델 설정 (model)
| 파라미터 | 설명 | 기본값 | 비고 |
|----------|------|---------|------|
| name | 모델 이름 | "bert-base-uncased" | Hugging Face 모델 ID |
| task | 모델 태스크 | "Text2Text Generation" | 모델의 주 태스크 |
| output_dir | 모델 저장 경로 | "./models/fine_tuned_model" | 파인튜닝된 모델이 저장될 위치 |
| cache_dir | 캐시 저장 경로 | "./models/pretrained" | 사전학습 모델 캐시 위치 |

#### 2. 데이터 설정 (data)
| 파라미터 | 설명 | 기본값 | 비고 |
|----------|------|---------|------|
| dataset_name | 데이터셋 이름 | "imdb" | Hugging Face 데이터셋 ID |
| dataset_cache_dir | 데이터셋 캐시 경로 | "./datasets" | 데이터셋 캐시 저장 위치 |
| text_column | 텍스트 열 이름 | "text" | 입력 텍스트 열 |
| label_column | 레이블 열 이름 | "label" | 출력 레이블 열 |
| max_length | 최대 시퀀스 길이 | 1024 | 토큰 시퀀스 최대 길이 |
| batch_size | 배치 크기 | 16 | 학습 배치 크기 |

#### 3. 학습 설정 (training)
| 파라미터 | 설명 | 기본값 | 비고 |
|----------|------|---------|------|
| epochs | 학습 에포크 수 | 3 | 전체 데이터셋 반복 횟수 |
| learning_rate | 학습률 | 2e-5 | 모델 파라미터 업데이트 속도 |
| weight_decay | 가중치 감쇠 | 0.01 | 과적합 방지를 위한 정규화 |
| warmup_steps | 워밍업 스텝 수 | 0 | 학습률 점진적 증가 단계 |
| max_grad_norm | 최대 그래디언트 노름 | 1.0 | 그래디언트 클리핑 임계값 |

#### 4. 메모리 설정 (memory)
| 파라미터 | 설명 | 기본값 | 비고 |
|----------|------|---------|------|
| gradient_accumulation_steps | 그래디언트 누적 스텝 수 | 4 | 메모리 사용량 감소에 도움 |
| max_gpu_memory_usage | GPU 메모리 사용 제한 | 0.95 | 최대 GPU 메모리 사용률 |
| mixed_precision | 혼합 정밀도 학습 | true | 메모리 효율성 향상 |
| optimize_memory_usage | 메모리 최적화 | true | 추가 메모리 최적화 |

## 학습 과정

### 1. 학습 시작
```bash
python finetuning_app.py --config config.yaml
```

### 2. 학습 모니터링
- 진행 상황이 실시간으로 출력됩니다
- 에포크별 손실값과 정확도가 표시됩니다
- GPU 메모리 사용량이 주기적으로 보고됩니다

### 3. 학습 중단 및 재개
- Ctrl+C로 안전하게 중단 가능
- 중단 시 현재 상태가 자동 저장됨
- `--resume` 플래그로 학습 재개 가능

## 문제 해결

### 1. 메모리 관련 문제
- **증상**: CUDA out of memory 에러
- **해결방법**:
  ```yaml
  # config.yaml
  data:
    batch_size: 8  # 기본값보다 감소
    max_length: 512  # 기본값보다 감소
  
  memory:
    gradient_accumulation_steps: 8  # 기본값보다 증가
    mixed_precision: true
  ```

### 2. 데이터셋 로드 실패
- **증상**: 데이터셋을 찾을 수 없음
- **해결방법**:
  - 인터넷 연결 확인
  - dataset_name 철자 확인
  - Hugging Face 로그인 상태 확인

### 3. 모델 다운로드 실패
- **증상**: 모델을 찾을 수 없음
- **해결방법**:
  - model_name 철자 확인
  - 인터넷 연결 확인
  - Hugging Face 토큰 설정:
    ```bash
    huggingface-cli login
    ```

## FAQ

### Q: 최소 필요 GPU 메모리는?
A: 모델 크기에 따라 다르지만, 일반적으로:
- 기본 설정: 16GB VRAM
- 메모리 최적화 설정: 8GB VRAM
- CPU만 사용: 32GB RAM

### Q: 학습 시간은 얼마나 걸리나요?
A: 데이터셋 크기, GPU 성능, 설정에 따라 다릅니다:
- 작은 데이터셋 (1만 문장): 2-3시간
- 중간 데이터셋 (5만 문장): 10-12시간
- 큰 데이터셋 (10만 문장 이상): 24시간 이상

### Q: 중간에 학습이 중단되면?
A: 자동 저장된 체크포인트에서 재개할 수 있습니다:
```bash
python finetuning_app.py --resume --checkpoint_path "./models/checkpoint-latest"
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 LICENSE 파일을 참조하세요.

## 기여하기

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

## 연락처

프로젝트 관리자: [이름]
이메일: [이메일 주소] 