"""데이터 관리 모듈

이 모듈은 텍스트 분류 작업을 위한 데이터셋의 로드, 전처리, 검증을 담당합니다.
데이터셋을 로드하고, 토큰화하며, DataLoader를 생성하는 기능을 제공합니다.

주요 기능:
- 데이터셋 로드 및 전처리
- 데이터 검증
- 토큰화
- DataLoader 생성

일반적인 사용 예:
```python
data_manager = DataManager(config, tokenizer)
train_dataloader = data_manager.load_and_preprocess_data("train")
test_dataloader = data_manager.load_and_preprocess_data("test")
```

Classes:
    TextDataset: PyTorch Dataset 구현체
    DataManager: 데이터 관리 클래스
"""

import logging
from typing import Dict, Any, Tuple, List, Union
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizer
from data_validation import DataValidator
from exceptions import DatasetError

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """텍스트 데이터셋 클래스
    
    토큰화된 텍스트 데이터와 레이블을 PyTorch Dataset 형태로 제공합니다.
    
    Attributes:
        encodings (Dict[str, list]): 토큰화된 입력 데이터
        labels (list): 레이블 데이터
    """
    
    def __init__(self, encodings: Dict[str, list], labels: list) -> None:
        """TextDataset 초기화
        
        Args:
            encodings: 토큰화된 입력 데이터
                필수 키:
                - input_ids: 입력 토큰 ID 리스트
                - attention_mask: 어텐션 마스크 리스트
            labels: 레이블 데이터 리스트
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """인덱스에 해당하는 데이터 샘플 반환
        
        Args:
            idx: 데이터 인덱스
            
        Returns:
            Dict[str, torch.Tensor]: 텐서로 변환된 데이터 샘플
                포함 키:
                - input_ids: 입력 토큰 ID 텐서
                - attention_mask: 어텐션 마스크 텐서
                - labels: 레이블 텐서
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        """데이터셋의 총 샘플 수 반환
        
        Returns:
            int: 데이터셋의 총 샘플 수
        """
        return len(self.labels)

class DataManager:
    """데이터 관리 클래스
    
    텍스트 분류를 위한 데이터셋의 로드, 전처리, 검증을 담당하는 클래스입니다.
    데이터셋을 로드하고, 토큰화하며, 검증하고, DataLoader를 생성하는 기능을 제공합니다.
    
    Attributes:
        config (Dict[str, Any]): 데이터 관련 설정
        tokenizer (PreTrainedTokenizer): 토크나이저 인스턴스
        dataset_name (str): 데이터셋 이름
        text_column (str): 텍스트 데이터 컬럼명
        label_column (str): 레이블 데이터 컬럼명
        max_length (int): 최대 시퀀스 길이
        batch_size (int): 배치 크기
        validator (DataValidator): 데이터 검증기 인스턴스
        
    Raises:
        DatasetError: 설정이 유효하지 않거나 필수 필드가 누락된 경우
    """
    
    def __init__(self, config: Dict[str, Any], tokenizer: PreTrainedTokenizer) -> None:
        """DataManager 초기화
        
        Args:
            config: 데이터 관련 설정
                필수 키:
                - dataset_name (str): 데이터셋 이름
                - text_column (str): 텍스트 컬럼명
                - label_column (str): 레이블 컬럼명
                - max_length (int): 최대 시퀀스 길이
                - batch_size (int): 배치 크기
            tokenizer: 토크나이저 인스턴스
            
        Raises:
            DatasetError: 설정이 유효하지 않거나 필수 필드가 누락된 경우
        """
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_name = config['dataset_name']
        self.text_column = config['text_column']
        self.label_column = config['label_column']
        self.max_length = config['max_length']
        self.batch_size = config['batch_size']
        
        # 데이터 검증기 초기화
        self.validator = DataValidator(config)
        
        # 설정 검증
        self.validator.validate_dataset_config()
    
    def load_and_preprocess_data(self, split: str = "train") -> DataLoader:
        """데이터셋을 로드하고 전처리
        
        데이터셋을 로드하고, 검증하고, 토큰화한 후 DataLoader를 생성합니다.
        
        Args:
            split: 데이터셋 분할("train" 또는 "test")
            
        Returns:
            DataLoader: 처리된 데이터 로더
                각 배치는 다음 키를 포함하는 딕셔너리:
                - input_ids: 입력 토큰 ID 텐서
                - attention_mask: 어텐션 마스크 텐서
                - labels: 레이블 텐서
            
        Raises:
            DatasetError: 데이터 처리 중 에러 발생 시
                - 데이터셋 로드 실패
                - 데이터 검증 실패
                - 토큰화 실패
                - DataLoader 생성 실패
        """
        dataset = self._load_dataset(split)
        return self._preprocess_data(dataset)
    
    def _load_dataset(self, split: str) -> Any:
        """데이터셋을 로드하고 검증
        
        Args:
            split: 데이터셋 분할("train" 또는 "test")
            
        Returns:
            로드된 데이터셋
            
        Raises:
            DatasetError: 다음 경우에 발생
                - 데이터셋 로드 실패
                - 데이터 검증 실패
                - 필수 필드 누락
        """
        try:
            logger.info(f"데이터셋 '{self.dataset_name}' ({split}) 로드 중...")
            dataset = load_dataset(
                self.dataset_name,
                split=split,
                cache_dir=self.config.get('dataset_cache_dir', './datasets')
            )
            
            # 데이터셋 통계 로깅
            self.validator.log_data_statistics(dataset)
            
            # 데이터 샘플 검증
            for idx, sample in enumerate(dataset):
                try:
                    self.validator.validate_data_sample(sample)
                except DatasetError as e:
                    logger.warning(f"샘플 {idx} 검증 실패: {str(e)}")
                    continue
            
            logger.info(f"데이터셋 로드 완료: {len(dataset)} 샘플")
            return dataset
            
        except Exception as e:
            error_msg = f"데이터셋 '{self.dataset_name}' 로드 중 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise DatasetError(error_msg) from e
    
    def _preprocess_data(self, dataset: Any) -> DataLoader:
        """데이터셋을 전처리하고 DataLoader 생성
        
        다음 단계를 수행:
        1. 데이터 토큰화
        2. 토큰화된 데이터 검증
        3. DataLoader 생성
        4. 배치 데이터 검증
        
        Args:
            dataset: 처리할 데이터셋
            
        Returns:
            DataLoader: 처리된 데이터 로더
        """
        try:
            logger.info("데이터 전처리 중...")
            logger.info(f"설정된 최대 길이: {self.max_length}")
            
            # 데이터 토큰화 및 검증
            def tokenize_and_validate(examples):
                # 토큰화 수행
                tokenized = self.tokenizer(
                    examples[self.text_column],
                    padding=True,  # 동적 패딩 사용
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors=None  # 배치 처리를 위해 리스트 형태 반환
                )
                
                try:
                    # 토큰화된 데이터 검증
                    self.validator.validate_tokenized_data(tokenized, self.max_length)
                except Exception as e:
                    logger.warning(f"토큰화 검증 중 경고: {str(e)}")
                
                return tokenized
            
            # 배치 크기 조정을 통한 토큰화
            # 레이블 컬럼을 제외한 컬럼만 제거
            columns_to_remove = [col for col in dataset.column_names if col != self.label_column]
            
            tokenized_dataset = dataset.map(
                tokenize_and_validate,
                batched=True,
                batch_size=100,  # 작은 배치 크기로 처리
                desc="토큰화 중...",
                remove_columns=columns_to_remove  # 레이블 컬럼은 유지
            )
            
            # 필요한 키만 선택
            tokenized_dict = {
                'input_ids': tokenized_dataset['input_ids'],
                'attention_mask': tokenized_dataset['attention_mask']
            }
            
            # DataLoader 생성
            dataset_tensor = TextDataset(
                tokenized_dict,
                tokenized_dataset[self.label_column]
            )
            
            dataloader = DataLoader(
                dataset_tensor,
                batch_size=self.batch_size,
                shuffle=(dataset.split == "train"),
                num_workers=2,
                pin_memory=True
            )
            
            # 첫 번째 배치 검증
            sample_batch = next(iter(dataloader))
            self.validator.validate_batch(sample_batch)
            
            logger.info(f"데이터 전처리 완료: {len(dataloader)} 배치")
            return dataloader
            
        except Exception as e:
            error_msg = f"데이터 전처리 중 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise DatasetError(error_msg) from e 