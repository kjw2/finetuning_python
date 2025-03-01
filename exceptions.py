"""커스텀 예외 클래스 모듈

이 모듈은 애플리케이션에서 사용되는 커스텀 예외 클래스들을 정의합니다.
"""

class FineTuningError(Exception):
    """파인튜닝 중 발생하는 기본 예외 클래스"""
    pass

class DatasetError(FineTuningError):
    """데이터셋 처리 중 발생하는 예외"""
    pass

class ModelError(Exception):
    """모델 관련 작업 중 발생하는 예외"""
    pass

class GPUMemoryError(Exception):
    """GPU 메모리 관련 예외"""
    pass

class ConfigError(FineTuningError):
    """설정 관련 예외"""
    pass

class TrainingError(FineTuningError):
    """학습 중 발생하는 예외"""
    pass

class DataError(Exception):
    """데이터 처리 중 발생하는 예외"""
    pass 