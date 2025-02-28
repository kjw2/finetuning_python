"""모델 관리 모듈

이 모듈은 모델의 로드, 저장, 초기화를 담당합니다.
"""

import os
import logging
from typing import Dict, Any, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from exceptions import ModelError

logger = logging.getLogger(__name__)

class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 모델 관련 설정
        """
        self.config = config
        if not isinstance(config, dict):
            raise ModelError("모델 설정이 딕셔너리 형태가 아닙니다")
            
        self.model_name = config.get('name')
        if not self.model_name:
            raise ModelError("모델 이름이 설정되지 않았습니다")
            
        self.output_dir = config.get('output_dir')
        if not self.output_dir:
            raise ModelError("모델 출력 디렉토리가 설정되지 않았습니다")
            
        self.task = config.get('task', 'text-classification')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.tokenizer = None
    
    def setup(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """모델과 토크나이저 설정
        
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: 모델과 토크나이저
            
        Raises:
            ModelError: 모델 초기화 중 에러 발생 시
        """
        try:
            # 출력 디렉토리 생성
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 캐시 디렉토리 생성
            cache_dir = self.config.get('cache_dir', './models/pretrained')
            os.makedirs(cache_dir, exist_ok=True)
            
            logger.info(f"디바이스: {self.device}")
            logger.info(f"모델 '{self.model_name}' 다운로드 중...")
            
            # 토크나이저와 모델 로드
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=cache_dir
            )
            self.model.to(self.device)
            
            logger.info(f"모델과 토크나이저 로드 완료")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            error_msg = f"모델 초기화 중 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    def save_model(self) -> None:
        """모델 저장"""
        try:
            logger.info(f"모델을 '{self.output_dir}'에 저장 중...")
            self.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info("모델과 토크나이저 저장 완료")
            
        except Exception as e:
            error_msg = f"모델 저장 중 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e
    
    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """저장된 모델 로드
        
        Returns:
            Tuple[PreTrainedModel, PreTrainedTokenizer]: 모델과 토크나이저
            
        Raises:
            ModelError: 모델 로드 중 에러 발생 시
        """
        try:
            logger.info(f"저장된 모델을 '{self.output_dir}'에서 로드 중...")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
            self.model.to(self.device)
            
            logger.info("저장된 모델과 토크나이저 로드 완료")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            error_msg = f"저장된 모델 로드 중 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise ModelError(error_msg) from e 