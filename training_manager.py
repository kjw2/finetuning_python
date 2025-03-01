import logging
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm
from memory_utils import MemoryManager
from exceptions import FineTuningError

logger = logging.getLogger(__name__)

class TrainingManager:
    """학습 관리 클래스"""
    
    def __init__(self, model: PreTrainedModel, config: Dict[str, Any],
                 memory_manager: MemoryManager, device: torch.device):
        """
        Args:
            model: 학습할 모델
            config: 학습 관련 설정
            memory_manager: 메모리 관리자
            device: 학습 디바이스
        """
        self.model = model
        self.config = config
        self.memory_manager = memory_manager
        self.device = device
        
        self.epochs = config['epochs']
        self.learning_rate = float(config['learning_rate'])
        self.weight_decay = config['weight_decay']
        self.warmup_steps = config['warmup_steps']
        self.max_grad_norm = config['max_grad_norm']
    
    def train(self, train_dataloader: DataLoader) -> None:
        """모델 학습 수행
        
        Args:
            train_dataloader: 학습 데이터 로더
            
        Raises:
            FineTuningError: 학습 중 에러 발생 시
        """
        logger.info("학습 시작...")
        
        try:
            # 옵티마이저 설정
            optimizer = self._setup_optimizer()
            
            # 학습 스케줄러
            scheduler = self._setup_scheduler(train_dataloader)
            
            # 학습 루프
            self.model.train()
            for epoch in range(self.epochs):
                try:
                    self._train_epoch(epoch, train_dataloader, optimizer, scheduler)
                    
                except KeyboardInterrupt:
                    logger.warning("사용자가 학습을 중단했습니다.")
                    raise
                    
                except Exception as e:
                    error_msg = f"에포크 {epoch + 1} 학습 중 오류 발생: {str(e)}"
                    logger.error(error_msg)
                    raise FineTuningError(error_msg) from e
        
        except Exception as e:
            error_msg = f"학습 중 치명적인 오류 발생: {str(e)}"
            logger.error(error_msg)
            raise FineTuningError(error_msg) from e
        
        logger.info("학습 완료")
    
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        """모델 평가 수행
        
        Args:
            eval_dataloader: 평가 데이터 로더
            
        Returns:
            float: 정확도
            
        Raises:
            FineTuningError: 평가 중 에러 발생 시
        """
        try:
            logger.info("모델 평가 중...")
            self.model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(eval_dataloader, desc="평가 중"):
                    try:
                        if not batch:
                            logger.warning("빈 배치를 건너뜁니다.")
                            continue
                            
                        metrics = self._evaluate_batch(batch)
                        correct += metrics['correct']
                        total += metrics['total']
                        
                    except Exception as e:
                        error_msg = f"배치 평가 중 오류 발생: {str(e)}"
                        logger.error(error_msg)
                        continue
            
            if total == 0:
                error_msg = "평가할 데이터가 없습니다. 데이터셋과 데이터 로더를 확인해주세요."
                logger.error(error_msg)
                raise FineTuningError(error_msg)
                
            accuracy = correct / total
            logger.info(f"평가 완료 - 정확도: {accuracy:.4f} (맞은 개수: {correct}, 전체 개수: {total})")
            return accuracy
            
        except Exception as e:
            error_msg = f"평가 중 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise FineTuningError(error_msg) from e
    
    def _setup_optimizer(self) -> AdamW:
        """옵티마이저 설정"""
        return AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
    
    def _setup_scheduler(self, train_dataloader: DataLoader):
        """학습 스케줄러 설정"""
        total_steps = len(train_dataloader) * self.epochs
        return get_linear_schedule_with_warmup(
            self._setup_optimizer(),
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
    
    def _train_epoch(self, epoch: int, train_dataloader: DataLoader,
                    optimizer: AdamW, scheduler) -> None:
        """한 에포크 학습
        
        Args:
            epoch: 현재 에포크
            train_dataloader: 학습 데이터 로더
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러
        """
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 메모리 최적화
                self.memory_manager.optimize_memory(batch_idx)
                
                # 배치 학습
                loss = self._train_batch(batch, optimizer, scheduler, batch_idx)
                epoch_loss += loss
                
                # 진행 상황 업데이트
                progress_bar.set_postfix({
                    "loss": loss * self.memory_manager.gradient_accumulation_steps
                })
                
            except Exception as e:
                error_msg = f"배치 {batch_idx} 처리 중 오류 발생: {str(e)}"
                logger.error(error_msg)
                continue
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"에포크 {epoch+1}/{self.epochs} 완료 - 평균 손실: {avg_epoch_loss:.4f}")
    
    def _train_batch(self, batch: Dict[str, torch.Tensor], optimizer: AdamW,
                    scheduler, batch_idx: int) -> float:
        """배치 학습
        
        Args:
            batch: 학습 배치
            optimizer: 옵티마이저
            scheduler: 학습률 스케줄러
            batch_idx: 배치 인덱스
            
        Returns:
            float: 배치 손실값
        """
        # 데이터를 GPU로 이동
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 혼합 정밀도 컨텍스트에서 순전파
        with self.memory_manager.autocast_context():
            outputs = self.model(**batch)
            loss = outputs.loss / self.memory_manager.gradient_accumulation_steps
        
        # 역전파
        self.memory_manager.backward_pass(loss, batch_idx)
        
        # 그래디언트 누적이 완료되면 옵티마이저 스텝
        if not self.memory_manager.should_accumulate_gradient(batch_idx):
            self.memory_manager.optimizer_step(optimizer, self.model, self.max_grad_norm)
            scheduler.step()
        
        return loss.item()
    
    def _evaluate_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """배치 평가
        
        Args:
            batch: 평가할 배치 데이터
            
        Returns:
            Dict[str, int]: 평가 메트릭 (correct: 맞은 개수, total: 전체 개수)
        """
        # 데이터를 GPU로 이동
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # 예측 수행
        outputs = self.model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        # 정확도 계산
        labels = batch['labels']
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        
        if total == 0:
            logger.warning("현재 배치의 크기가 0입니다.")
            return {'correct': 0, 'total': 0}
            
        return {'correct': correct, 'total': total} 