"""데이터 검증을 위한 유틸리티 모듈

이 모듈은 데이터셋, 데이터 샘플, 토큰화된 데이터, 배치 데이터에 대한
포괄적인 검증 기능을 제공합니다.

주요 기능:
- 데이터셋 설정 검증
- 개별 데이터 샘플 검증
- 토큰화된 데이터 검증
- 배치 데이터 검증
- 데이터셋 통계 분석 및 로깅

일반적인 사용 예:
```python
validator = DataValidator(config)
validator.validate_dataset_config()
validator.validate_data_sample(sample)
validator.validate_tokenized_data(tokenized_data, max_length)
validator.validate_batch(batch)
validator.log_data_statistics(dataset)
```

Raises:
    DatasetError: 데이터 검증 과정에서 발생하는 모든 예외의 기본 클래스
"""

from typing import Dict, Any, List, Optional, Union, Sequence
import logging
from exceptions import DatasetError

logger = logging.getLogger(__name__)

class DataValidator:
    """데이터 검증을 위한 클래스
    
    이 클래스는 텍스트 분류 작업을 위한 데이터셋의 무결성과 품질을 보장하기 위한
    다양한 검증 메서드를 제공합니다.
    
    Attributes:
        config (Dict[str, Any]): 데이터 관련 설정
        text_column (str): 텍스트 데이터가 포함된 컬럼 이름
        label_column (str): 레이블 데이터가 포함된 컬럼 이름
        max_length (int): 최대 시퀀스 길이
        
    Raises:
        DatasetError: 설정이 유효하지 않거나 필수 필드가 누락된 경우
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """DataValidator 초기화
        
        Args:
            config: 데이터 관련 설정이 포함된 딕셔너리
                필수 키:
                - dataset_name (str): 데이터셋 이름
                - text_column (str): 텍스트 컬럼 이름
                - label_column (str): 레이블 컬럼 이름
                - max_length (int): 최대 시퀀스 길이
                - batch_size (int): 배치 크기
                
        Raises:
            DatasetError: 필수 설정이 누락되거나 유효하지 않은 경우
        """
        self.config = config
        self.validate_dataset_config()  # 설정 검증 먼저 수행
        self.text_column = config['text_column']
        self.label_column = config['label_column']
        self.max_length = config['max_length']
    
    def validate_dataset_config(self) -> None:
        """데이터셋 설정의 유효성을 검증
        
        다음 항목을 검증:
        1. 모든 필수 설정 필드의 존재 여부
        2. max_length가 양의 정수인지
        3. batch_size가 양의 정수인지
        
        Raises:
            DatasetError: 다음 경우에 발생
                - 필수 설정 필드가 누락된 경우
                - max_length가 양의 정수가 아닌 경우
                - batch_size가 양의 정수가 아닌 경우
        """
        required_fields = ['dataset_name', 'text_column', 'label_column', 'max_length', 'batch_size']
        for field in required_fields:
            if field not in self.config:
                raise DatasetError(f"필수 설정 필드 누락: {field}")
        
        if not isinstance(self.config['max_length'], int) or self.config['max_length'] <= 0:
            raise DatasetError("max_length는 양의 정수여야 합니다")
            
        if not isinstance(self.config['batch_size'], int) or self.config['batch_size'] <= 0:
            raise DatasetError("batch_size는 양의 정수여야 합니다")
    
    def validate_data_sample(self, sample: Dict[str, Any]) -> None:
        """개별 데이터 샘플의 유효성을 검증
        
        다음 항목을 검증:
        1. 필수 필드(텍스트, 레이블) 존재 여부
        2. 텍스트 데이터의 타입 및 내용
        3. 레이블 데이터의 타입
        
        Args:
            sample: 검증할 데이터 샘플
                필수 키:
                - text_column에 해당하는 키: 문자열 텍스트
                - label_column에 해당하는 키: 숫자 레이블
                
        Raises:
            DatasetError: 다음 경우에 발생
                - 필수 필드가 누락된 경우
                - 텍스트가 문자열이 아닌 경우
                - 텍스트가 비어있는 경우
                - 레이블이 숫자가 아닌 경우
        """
        # 필수 필드 존재 확인
        if self.text_column not in sample:
            raise DatasetError(f"텍스트 컬럼 '{self.text_column}' 누락")
        if self.label_column not in sample:
            raise DatasetError(f"레이블 컬럼 '{self.label_column}' 누락")
        
        # 텍스트 데이터 검증
        text = sample[self.text_column]
        if not isinstance(text, str):
            raise DatasetError(f"텍스트 데이터는 문자열이어야 합니다: {type(text)}")
        if not text.strip():
            raise DatasetError("빈 텍스트 데이터")
            
        # 레이블 데이터 검증
        label = sample[self.label_column]
        if not isinstance(label, (int, float)):
            raise DatasetError(f"레이블은 숫자여야 합니다: {type(label)}")
    
    def validate_tokenized_data(self, tokenized_data: Dict[str, List[int]], max_length: int) -> None:
        """토큰화된 데이터의 유효성을 검증
        
        Args:
            tokenized_data: 검증할 토큰화된 데이터
            max_length: 최대 허용 시퀀스 길이
        """
        required_fields = ['input_ids', 'attention_mask']
        for field in required_fields:
            if field not in tokenized_data:
                raise DatasetError(f"토큰화된 데이터에서 필수 필드 누락: {field}")
        
        for field in required_fields:
            data = tokenized_data[field]
            if not isinstance(data, list):
                raise DatasetError(f"토큰화된 데이터 필드는 리스트여야 합니다: {field}")
            
            if len(data) > max_length:
                logger.warning(f"토큰화된 데이터가 최대 길이를 초과하여 잘립니다: {len(data)} -> {max_length}")
    
    def validate_batch(self, batch: Dict[str, Any]) -> None:
        """배치 데이터의 유효성을 검증
        
        다음 항목을 검증:
        1. 필수 필드(input_ids, attention_mask, labels) 존재 여부
        2. 배치가 비어있지 않은지 확인
        3. 모든 필드의 배치 크기가 동일한지 확인
        4. input_ids와 attention_mask가 2차원 텐서인지 확인
        
        Args:
            batch: 검증할 배치 데이터
                필수 키:
                - input_ids: 입력 토큰 ID 텐서
                - attention_mask: 어텐션 마스크 텐서
                - labels: 레이블 텐서
                
        Raises:
            DatasetError: 다음 경우에 발생
                - 필수 필드가 누락된 경우
                - 배치가 비어있는 경우
                - 필드 간 배치 크기가 일치하지 않는 경우
                - input_ids나 attention_mask가 2차원 텐서가 아닌 경우
        """
        required_fields = ['input_ids', 'attention_mask', 'labels']
        for field in required_fields:
            if field not in batch:
                raise DatasetError(f"배치 데이터에서 필수 필드 누락: {field}")
        
        # input_ids와 attention_mask가 2차원 텐서인지 확인
        for field in ['input_ids', 'attention_mask']:
            if batch[field].dim() != 2:
                raise DatasetError(f"배치 크기 불일치: {field}는 2차원 텐서여야 합니다")
        
        # 배치 크기 검증
        batch_size = batch['input_ids'].size(0)
        if batch_size == 0:
            raise DatasetError("빈 배치 데이터")
        
        # 모든 필드의 배치 크기가 동일한지 확인
        for field in required_fields:
            if batch[field].size(0) != batch_size:
                raise DatasetError(f"배치 크기 불일치: {field}의 크기가 {batch[field].size(0)}이지만 예상값은 {batch_size}")
    
    def log_data_statistics(self, dataset: Sequence[Dict[str, Any]]) -> None:
        """데이터셋의 통계 정보를 계산하고 로깅
        
        다음 통계를 계산하고 로깅:
        1. 총 샘플 수
        2. 텍스트 길이 통계 (평균, 최대, 최소)
        3. 레이블 분포
        
        Args:
            dataset: 통계를 계산할 데이터셋
                각 샘플은 딕셔너리 형태로 text_column과 label_column을 포함해야 함
        
        Note:
            이 메서드는 예외를 발생시키지 않으며, 오류 발생 시 경고 메시지를 로깅
        """
        try:
            total_samples = len(dataset)
            text_lengths = [len(str(sample[self.text_column])) for sample in dataset]
            avg_length = sum(text_lengths) / total_samples
            max_length = max(text_lengths)
            min_length = min(text_lengths)
            
            logger.info(f"데이터셋 통계:")
            logger.info(f"- 총 샘플 수: {total_samples}")
            logger.info(f"- 평균 텍스트 길이: {avg_length:.2f}")
            logger.info(f"- 최대 텍스트 길이: {max_length}")
            logger.info(f"- 최소 텍스트 길이: {min_length}")
            
            # 레이블 분포 계산
            label_counts = {}
            for sample in dataset:
                label = sample[self.label_column]
                label_counts[label] = label_counts.get(label, 0) + 1
            
            logger.info("- 레이블 분포:")
            for label, count in label_counts.items():
                percentage = (count / total_samples) * 100
                logger.info(f"  레이블 {label}: {count} ({percentage:.2f}%)")
                
        except Exception as e:
            logger.warning(f"데이터셋 통계 계산 중 오류 발생: {str(e)}") 