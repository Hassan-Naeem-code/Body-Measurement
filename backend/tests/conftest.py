"""
Pytest Fixtures for FitWhisperer Tests
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.core.database import Base, get_db
from app.models.brand import Brand, SubscriptionTier
from app.models.product import Product, SizeChart
from app.core.auth import hash_password
import secrets


# Test database setup
SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for tests"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """Create a test client with database override"""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_brand(db_session):
    """Create a test brand with API key"""
    api_key = f"test_{secrets.token_urlsafe(32)}"
    brand = Brand(
        name="Test Brand",
        email="test@example.com",
        password_hash=hash_password("TestPass123!"),
        api_key=api_key,
        is_active=True,
        subscription_tier=SubscriptionTier.FREE
    )
    db_session.add(brand)
    db_session.commit()
    db_session.refresh(brand)
    return brand


@pytest.fixture
def auth_headers(test_brand):
    """Return headers with API key for authenticated requests"""
    return {"X-API-Key": test_brand.api_key}


@pytest.fixture
def test_product(db_session, test_brand):
    """Create a test product with size charts"""
    product = Product(
        brand_id=test_brand.id,
        name="Test T-Shirt",
        category="tops",
        subcategory="t-shirt",
        gender="male",
        age_group="adult",
        is_active=True
    )
    db_session.add(product)
    db_session.flush()

    # Add size charts
    sizes = [
        {"size_name": "S", "chest_min": 86, "chest_max": 91, "waist_min": 71, "waist_max": 76},
        {"size_name": "M", "chest_min": 91, "chest_max": 99, "waist_min": 76, "waist_max": 84},
        {"size_name": "L", "chest_min": 99, "chest_max": 107, "waist_min": 84, "waist_max": 94},
    ]

    for i, size in enumerate(sizes):
        chart = SizeChart(
            product_id=product.id,
            size_name=size["size_name"],
            chest_min=size["chest_min"],
            chest_max=size["chest_max"],
            waist_min=size["waist_min"],
            waist_max=size["waist_max"],
            fit_type="regular",
            display_order=i
        )
        db_session.add(chart)

    db_session.commit()
    db_session.refresh(product)
    return product


@pytest.fixture
def sample_measurements():
    """Sample body measurements for testing"""
    return {
        "chest_circumference": 95.0,
        "waist_circumference": 80.0,
        "hip_circumference": 98.0,
        "height": 175.0
    }
