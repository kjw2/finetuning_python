"""메모리 관리 유틸리티 모듈

이 모듈은 GPU 메모리 사용을 최적화하고 관리하는 기능을 제공합니다.

주요 기능:
- GPU 메모리 사용량 모니터링
- 그래디언트 누적을 통한 메모리 최적화
- 혼합 정밀도 학습 지원
- CUDA 캐시 관리
"""

import gc
import logging
import torch
from typing import Dict, Any, Optional, ContextManager
from contextlib import contextmanager
from exceptions import GPUMemoryError

logger = logging.getLogger(__name__)

class MemoryManager:
    """메모리 관리 클래스
    
    GPU 메모리 사용을 최적화하고 관리하는 클래스입니다.
    그래디언트 누적, 혼합 정밀도 학습, CUDA 캐시 관리 등의 기능을 제공합니다.
    
    Attributes:
        config (Dict[str, Any]): 메모리 관리 관련 설정
        gradient_accumulation_steps (int): 그래디언트 누적 스텝 수
        max_gpu_memory_usage (float): 최대 GPU 메모리 사용률
        empty_cache_freq (int): CUDA 캐시 비우기 빈도
        mixed_precision (bool): 혼합 정밀도 학습 사용 여부
        optimize_memory_usage (bool): 메모리 최적화 사용 여부
        scaler (torch.cuda.amp.GradScaler): 그래디언트 스케일러
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """MemoryManager 초기화
        
        Args:
            config: 메모리 관리 관련 설정
                필수 키:
                - gradient_accumulation_steps (int): 그래디언트 누적 스텝 수
                - max_gpu_memory_usage (float): 최대 GPU 메모리 사용률 (0~1)
                - empty_cache_freq (int): CUDA 캐시 비우기 빈도
                - mixed_precision (bool): 혼합 정밀도 학습 사용 여부
                - optimize_memory_usage (bool): 메모리 최적화 사용 여부
        """
        self.config = config
        self.gradient_accumulation_steps = config['gradient_accumulation_steps']
        self.max_gpu_memory_usage = config['max_gpu_memory_usage']
        self.empty_cache_freq = config['empty_cache_freq']
        self.mixed_precision = config['mixed_precision']
        self.optimize_memory_usage = config['optimize_memory_usage']
        
        # 혼합 정밀도 학습을 위한 스케일러 초기화
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        logger.info(f"메모리 관리자 초기화 완료:")
        logger.info(f"- 그래디언트 누적 스텝: {self.gradient_accumulation_steps}")
        logger.info(f"- 최대 GPU 메모리 사용률: {self.max_gpu_memory_usage}")
        logger.info(f"- CUDA 캐시 비우기 빈도: {self.empty_cache_freq}")
        logger.info(f"- 혼합 정밀도 학습: {self.mixed_precision}")
        logger.info(f"- 메모리 최적화: {self.optimize_memory_usage}")
    
    def optimize_memory(self, batch_idx: int) -> None:
        """메모리 최적화 수행
        
        Args:
            batch_idx: 현재 배치 인덱스
        """
        if not self.optimize_memory_usage:
            return
            
        # CUDA 캐시 비우기
        if batch_idx % self.empty_cache_freq == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
        # GPU 메모리 사용량 체크
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_allocated > self.max_gpu_memory_usage:
                logger.warning(f"GPU 메모리 사용량이 임계값을 초과: {memory_allocated:.2%}")
                torch.cuda.empty_cache()
                gc.collect()
    
    @contextmanager
    def autocast_context(self) -> ContextManager:
        """혼합 정밀도 학습을 위한 컨텍스트 매니저
        
        Returns:
            ContextManager: 혼합 정밀도 컨텍스트
        """
        if self.mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                yield
        else:
            yield
    
    def backward_pass(self, loss: torch.Tensor, batch_idx: int) -> None:
        """역전파 수행
        
        Args:
            loss: 손실값
            batch_idx: 현재 배치 인덱스
        """
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer,
                      model: torch.nn.Module,
                      max_grad_norm: float) -> None:
        """옵티마이저 스텝 수행
        
        Args:
            optimizer: 옵티마이저 인스턴스
            model: 모델 인스턴스
            max_grad_norm: 최대 그래디언트 노름
        """
        if self.mixed_precision:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        optimizer.zero_grad()
    
    def should_accumulate_gradient(self, batch_idx: int) -> bool:
        """그래디언트를 누적해야 하는지 확인
        
        Args:
            batch_idx: 현재 배치 인덱스
            
        Returns:
            bool: 그래디언트를 누적해야 하면 True
        """
        return (batch_idx + 1) % self.gradient_accumulation_steps != 0
    
    def get_memory_stats(self) -> Dict[str, float]:
        """메모리 사용 통계 반환
        
        Returns:
            Dict[str, float]: 메모리 사용 통계
                - allocated: 할당된 메모리 (GB)
                - cached: 캐시된 메모리 (GB)
                - peak: 최대 사용 메모리 (GB)
        """
        if not torch.cuda.is_available():
            return {'allocated': 0, 'cached': 0, 'peak': 0}
            
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'cached': torch.cuda.memory_reserved() / 1e9,
            'peak': torch.cuda.max_memory_allocated() / 1e9
        } 