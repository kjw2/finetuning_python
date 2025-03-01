import os
import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """설정 관리 클래스"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path (str): 설정 파일 경로
        """
        # 기본 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        if config_path is None:
            config_path = "config.yaml"
            logger.warning(f"config_path가 None입니다. 기본값 '{config_path}'를 사용합니다.")
        
        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(config_path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, config_path)
        
        self.config_path = os.path.abspath(config_path)
        logger.info(f"설정 파일 경로: {self.config_path}")
        
        self.config = self._load_config()
        
        # 설정 파일의 로깅 설정으로 업데이트
        self._setup_logging()
    
    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            if not os.path.exists(self.config_path):
                error_msg = f"설정 파일이 존재하지 않습니다: {self.config_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if not config:
                error_msg = "설정 파일이 비어있습니다"
                logger.error(error_msg)
                raise yaml.YAMLError(error_msg)
            
            # 설정 내용 출력 및 검증
            logger.info("=== 설정 파일 로드 디버깅 ===")
            logger.info(f"로드된 원본 설정: {config}")
            
            # 깊은 복사로 설정 복사
            validated_config = {}
            for section, values in config.items():
                logger.info(f"섹션 '{section}' 처리 중...")
                if values is None:
                    logger.warning(f"{section} 섹션의 값이 None입니다. 빈 딕셔너리로 초기화합니다.")
                    validated_config[section] = {}
                elif not isinstance(values, dict):
                    error_msg = f"{section} 섹션이 딕셔너리가 아닙니다: {values}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    # 깊은 복사 수행
                    validated_config[section] = {}
                    for key, value in values.items():
                        logger.info(f"  - {key}: {value}")
                        validated_config[section][key] = value
            
            # 필수 섹션 확인
            required_sections = ['model', 'data', 'training', 'memory', 'logging']
            for section in required_sections:
                if section not in validated_config:
                    error_msg = f"필수 섹션 '{section}'이 없습니다"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
            
            logger.info(f"검증된 최종 설정: {validated_config}")
            logger.info("=== 설정 파일 로드 완료 ===")
            return validated_config
            
        except yaml.YAMLError as e:
            error_msg = f"설정 파일 파싱 중 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise yaml.YAMLError(error_msg)
        except Exception as e:
            error_msg = f"설정 파일 로드 중 예상치 못한 에러 발생: {str(e)}"
            logger.error(error_msg)
            raise
    
    def _setup_logging(self) -> None:
        """로깅 설정"""
        logging_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, logging_config.get('level', 'INFO')),
            format=logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        )
    
    def get_model_config(self) -> Dict[str, Any]:
        """모델 관련 설정 반환"""
        logger.info("=== 모델 설정 디버깅 시작 ===")
        logger.info(f"전체 config: {self.config}")
        
        if 'model' not in self.config:
            raise ValueError("model 섹션이 설정에 없습니다")
        
        model_config = self.config['model']
        logger.info(f"원본 model_config: {model_config}")
        
        if not isinstance(model_config, dict):
            raise ValueError(f"model 섹션이 딕셔너리가 아닙니다: {model_config}")
        
        # 딕셔너리 깊은 복사
        result = {}
        for key, value in model_config.items():
            logger.info(f"복사 중: {key} = {value}")
            result[key] = value
        
        logger.info(f"최종 모델 설정: {result}")
        logger.info("=== 모델 설정 디버깅 종료 ===")
        return result
    
    def get_data_config(self) -> Dict[str, Any]:
        """데이터 관련 설정 반환"""
        logger.info("=== 데이터 설정 디버깅 시작 ===")
        logger.info(f"전체 config: {self.config}")
        
        if 'data' not in self.config:
            raise ValueError("data 섹션이 설정에 없습니다")
        
        data_config = self.config['data']
        logger.info(f"원본 data_config: {data_config}")
        
        if not isinstance(data_config, dict):
            raise ValueError(f"data 섹션이 딕셔너리가 아닙니다: {data_config}")
        
        # 딕셔너리 깊은 복사
        result = {}
        for key, value in data_config.items():
            logger.info(f"복사 중: {key} = {value}")
            result[key] = value
        
        logger.info(f"최종 데이터 설정: {result}")
        logger.info("=== 데이터 설정 디버깅 종료 ===")
        return result
    
    def get_training_config(self) -> Dict[str, Any]:
        """학습 관련 설정 반환"""
        logger.info("=== 학습 설정 디버깅 시작 ===")
        logger.info(f"전체 config: {self.config}")
        
        if 'training' not in self.config:
            raise ValueError("training 섹션이 설정에 없습니다")
        
        training_config = self.config['training']
        logger.info(f"원본 training_config: {training_config}")
        
        if not isinstance(training_config, dict):
            raise ValueError(f"training 섹션이 딕셔너리가 아닙니다: {training_config}")
        
        # 딕셔너리 깊은 복사
        result = {}
        for key, value in training_config.items():
            logger.info(f"복사 중: {key} = {value}")
            result[key] = value
        
        logger.info(f"최종 학습 설정: {result}")
        logger.info("=== 학습 설정 디버깅 종료 ===")
        return result
    
    def get_memory_config(self) -> Dict[str, Any]:
        """메모리 관리 관련 설정 반환"""
        logger.info("=== 메모리 설정 디버깅 시작 ===")
        logger.info(f"전체 config: {self.config}")
        
        if 'memory' not in self.config:
            raise ValueError("memory 섹션이 설정에 없습니다")
        
        memory_config = self.config['memory']
        logger.info(f"원본 memory_config: {memory_config}")
        
        if not isinstance(memory_config, dict):
            raise ValueError(f"memory 섹션이 딕셔너리가 아닙니다: {memory_config}")
        
        # 딕셔너리 깊은 복사
        result = {}
        for key, value in memory_config.items():
            logger.info(f"복사 중: {key} = {value}")
            result[key] = value
        
        logger.info(f"최종 메모리 설정: {result}")
        logger.info("=== 메모리 설정 디버깅 종료 ===")
        return result
    
    def update_from_args(self, args: Any) -> None:
        """커맨드 라인 인자로 설정 업데이트
        
        Args:
            args: argparse로 파싱된 인자들
        """
        # 모델 설정 업데이트
        if hasattr(args, 'model_name') and args.model_name is not None:
            self.config['model']['name'] = args.model_name
        if hasattr(args, 'output_dir') and args.output_dir is not None:
            self.config['model']['output_dir'] = args.output_dir
            
        # 데이터 설정 업데이트
        if hasattr(args, 'dataset') and args.dataset is not None:
            self.config['data']['dataset_name'] = args.dataset
        if hasattr(args, 'text_column') and args.text_column is not None:
            self.config['data']['text_column'] = args.text_column
        if hasattr(args, 'label_column') and args.label_column is not None:
            self.config['data']['label_column'] = args.label_column
        if hasattr(args, 'max_length') and args.max_length is not None:
            self.config['data']['max_length'] = args.max_length
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            self.config['data']['batch_size'] = args.batch_size
            
        # 학습 설정 업데이트
        if hasattr(args, 'epochs') and args.epochs is not None:
            self.config['training']['epochs'] = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate is not None:
            self.config['training']['learning_rate'] = float(args.learning_rate) 