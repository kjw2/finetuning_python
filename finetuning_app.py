import os
import argparse
import logging
from typing import Optional, Dict, Any
from config_manager import ConfigManager
from model_manager import ModelManager
from data_manager import DataManager
from training_manager import TrainingManager
from memory_utils import MemoryManager
from dataset import process_dataset
import yaml

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="모델 다운로드 및 파인튜닝 앱")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="설정 파일 경로")
    parser.add_argument("--model_name", type=str,
                        help="다운로드할 모델 이름 (Hugging Face 모델 ID)")
    parser.add_argument("--dataset", type=str,
                        help="사용할 데이터셋 이름")
    parser.add_argument("--output_dir", type=str,
                        help="파인튜닝된 모델을 저장할 디렉토리")
    parser.add_argument("--epochs", type=int,
                        help="파인튜닝 에포크 수")
    parser.add_argument("--batch_size", type=int,
                        help="배치 크기")
    parser.add_argument("--learning_rate", type=float,
                        help="학습률")
    parser.add_argument("--max_length", type=int,
                        help="최대 시퀀스 길이")
    parser.add_argument("--text_column", type=str,
                        help="데이터셋의 텍스트 열 이름")
    parser.add_argument("--label_column", type=str,
                        help="데이터셋의 레이블 열 이름")
    
    args = parser.parse_args()
    
    try:
        # 설정 관리자 초기화
        config_manager = ConfigManager(args.config)
        config_manager.update_from_args(args)
        
        # 설정값 로드 및 확인
        model_config = config_manager.get_model_config()
        data_config = config_manager.get_data_config()
        
        logger.info("=== 설정값 확인 ===")
        logger.info(f"모델 설정: {model_config}")
        logger.info(f"데이터 설정: {data_config}")
        
        # 데이터셋 처리
        processed_dataset = process_dataset(config_manager)
        
        # 메모리 관리자 초기화
        memory_manager = MemoryManager(config_manager.get_memory_config())
        
        # 모델 관리자 초기화 및 모델 설정
        model_manager = ModelManager(config_manager.get_model_config())
        model, tokenizer = model_manager.setup()
        
        # 데이터 관리자 초기화
        data_manager = DataManager(config_manager.get_data_config(), tokenizer)
        
        # 학습 데이터 로드 및 전처리
        train_dataloader = data_manager.load_and_preprocess_data(split="train")
        
        # 학습 관리자 초기화 및 학습 수행
        training_manager = TrainingManager(
            model=model,
            config=config_manager.get_training_config(),
            memory_manager=memory_manager,
            device=model_manager.device
        )
        
        try:
            # 모델 학습
            training_manager.train(train_dataloader)
            
            # 모델 저장
            model_manager.save_model()
            
            # 평가 데이터 로드 및 전처리
            eval_dataloader = data_manager.load_and_preprocess_data(split="test")
            
            # 모델 평가
            training_manager.evaluate(eval_dataloader)
            
        except KeyboardInterrupt:
            logger.warning("사용자가 작업을 중단했습니다. 현재 상태를 저장합니다.")
            model_manager.save_model()
        
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main()