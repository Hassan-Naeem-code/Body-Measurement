from pydantic_settings import BaseSettings
from typing import List, Union
from pydantic import field_validator


class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "Body Measurement API"
    API_V1_PREFIX: str = "/api/v1"
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = "postgresql://user:password@db:5432/body_measurement_db"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 1 week

    # CORS
    CORS_ORIGINS: Union[List[str], str] = "http://localhost:3000,http://localhost:3001"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # AI/ML Settings
    MAX_IMAGE_SIZE_MB: int = 10
    CONFIDENCE_THRESHOLD: float = 0.6
    DEFAULT_HEIGHT_CM: float = 170.0

    # YOLO Settings
    YOLO_MODEL_SIZE: str = "yolov8m.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5

    # Body Validation Settings
    # NOTE: EXTREMELY LOW thresholds for debugging (originally 0.6-0.7)
    # For production, increase these back to 0.6-0.7 for better quality
    BODY_VALIDATION_HEAD_THRESHOLD: float = 0.1
    BODY_VALIDATION_SHOULDERS_THRESHOLD: float = 0.1
    BODY_VALIDATION_ELBOWS_THRESHOLD: float = 0.1
    BODY_VALIDATION_HANDS_THRESHOLD: float = 0.1
    BODY_VALIDATION_TORSO_THRESHOLD: float = 0.1
    BODY_VALIDATION_LEGS_THRESHOLD: float = 0.1
    BODY_VALIDATION_FEET_THRESHOLD: float = 0.1  # Extremely low for debugging
    BODY_VALIDATION_OVERALL_MIN: float = 0.1  # Extremely low for debugging

    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


settings = Settings()
