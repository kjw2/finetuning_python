import os
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

def process_dataset(config_manager):
    """
    ConfigManager를 사용하여 데이터셋을 처리하는 함수
    
    Args:
        config_manager: ConfigManager 인스턴스
    """
    data_config = config_manager.get_data_config()
    model_config = config_manager.get_model_config()
    
    logger.info("데이터셋 로드 중...")
    dataset = load_dataset(
        data_config['dataset_name'],
        cache_dir=data_config['dataset_cache_dir']
    )

    logger.info("토크나이저 초기화 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name'],
        cache_dir=model_config['cache_dir']
    )

    logger.info("데이터셋 전처리 중...")
    return dataset.map(
        lambda x: tokenizer(
            x[data_config['text_column']],
            truncation=True,
            max_length=data_config['max_length']
        )
    )