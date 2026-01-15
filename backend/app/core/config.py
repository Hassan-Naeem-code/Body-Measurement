from pydantic_settings import BaseSettings
from typing import List, Union
from pydantic import field_validator
import secrets
import warnings


class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "FitWhisperer API"
    API_V1_PREFIX: str = "/api/v1"
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = "postgresql://user:password@db:5432/body_measurement_db"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Security - SECRET_KEY must be set in production via environment variable
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 1 week

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60  # requests per minute
    RATE_LIMIT_BURST: int = 10  # burst allowance

    # CORS
    CORS_ORIGINS: Union[List[str], str] = "http://localhost:3000,http://localhost:3001,http://localhost:8080"

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = False  # Default to False for safety

    @field_validator('SECRET_KEY', mode='before')
    @classmethod
    def validate_secret_key(cls, v, info):
        """Validate SECRET_KEY - generate for dev, require for production"""
        if not v or v == "your-secret-key-change-in-production" or v == "your-secret-key-here-change-this-in-production":
            # Check environment
            import os
            env = os.getenv('ENVIRONMENT', 'development')
            if env == 'production':
                raise ValueError(
                    "SECRET_KEY must be set in production! "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )
            # For development, generate a random key and warn
            generated_key = secrets.token_urlsafe(64)
            warnings.warn(
                "SECRET_KEY not set - using auto-generated key for development. "
                "Set SECRET_KEY environment variable for production!",
                UserWarning
            )
            return generated_key
        return v

    @field_validator('DEBUG', mode='before')
    @classmethod
    def validate_debug(cls, v, info):
        """Ensure DEBUG is False in production"""
        import os
        env = os.getenv('ENVIRONMENT', 'development')
        if env == 'production' and v in (True, 'true', 'True', '1', 1):
            warnings.warn(
                "DEBUG=True in production is dangerous! Setting to False.",
                UserWarning
            )
            return False
        # Convert string to bool if needed
        if isinstance(v, str):
            return v.lower() in ('true', '1', 'yes')
        return bool(v)

    # AI/ML Settings
    MAX_IMAGE_SIZE_MB: int = 10
    CONFIDENCE_THRESHOLD: float = 0.6
    DEFAULT_HEIGHT_CM: float = 170.0

    # YOLO Settings
    YOLO_MODEL_SIZE: str = "yolov8m.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5

    # Body Validation Settings
    # Production thresholds - balanced for accuracy while allowing real-world images
    BODY_VALIDATION_HEAD_THRESHOLD: float = 0.5
    BODY_VALIDATION_SHOULDERS_THRESHOLD: float = 0.55
    BODY_VALIDATION_ELBOWS_THRESHOLD: float = 0.4
    BODY_VALIDATION_HANDS_THRESHOLD: float = 0.35
    BODY_VALIDATION_TORSO_THRESHOLD: float = 0.5
    BODY_VALIDATION_LEGS_THRESHOLD: float = 0.45
    BODY_VALIDATION_FEET_THRESHOLD: float = 0.4
    BODY_VALIDATION_OVERALL_MIN: float = 0.45

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
