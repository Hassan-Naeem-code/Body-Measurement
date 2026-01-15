"""
Unit Tests for Pydantic Schemas
"""

import pytest
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.brand import BrandCreate, BrandLogin
from app.schemas.product import SizeChartCreate, SizeChartBase, ProductCreate


class TestBrandSchemas:
    """Tests for brand-related schemas"""

    def test_brand_create_valid(self):
        """Test valid brand creation"""
        brand = BrandCreate(
            name="Test Brand",
            email="test@example.com",
            password="ValidPass123!"
        )
        assert brand.name == "Test Brand"
        assert brand.email == "test@example.com"

    def test_brand_name_validation_empty(self):
        """Test brand name cannot be empty"""
        with pytest.raises(ValidationError) as exc_info:
            BrandCreate(
                name="",
                email="test@example.com",
                password="ValidPass123!"
            )
        assert "Brand name cannot be empty" in str(exc_info.value)

    def test_brand_name_validation_too_short(self):
        """Test brand name must be at least 2 characters"""
        with pytest.raises(ValidationError) as exc_info:
            BrandCreate(
                name="A",
                email="test@example.com",
                password="ValidPass123!"
            )
        assert "at least 2 characters" in str(exc_info.value)

    def test_password_too_short(self):
        """Test password must be at least 8 characters"""
        with pytest.raises(ValidationError) as exc_info:
            BrandCreate(
                name="Test Brand",
                email="test@example.com",
                password="Short1!"
            )
        assert "at least 8 characters" in str(exc_info.value)

    def test_password_no_uppercase(self):
        """Test password must have uppercase letter"""
        with pytest.raises(ValidationError) as exc_info:
            BrandCreate(
                name="Test Brand",
                email="test@example.com",
                password="nouppercase123!"
            )
        assert "uppercase" in str(exc_info.value)

    def test_password_no_lowercase(self):
        """Test password must have lowercase letter"""
        with pytest.raises(ValidationError) as exc_info:
            BrandCreate(
                name="Test Brand",
                email="test@example.com",
                password="NOLOWERCASE123!"
            )
        assert "lowercase" in str(exc_info.value)

    def test_password_no_digit(self):
        """Test password must have a digit"""
        with pytest.raises(ValidationError) as exc_info:
            BrandCreate(
                name="Test Brand",
                email="test@example.com",
                password="NoDigitsHere!"
            )
        assert "digit" in str(exc_info.value)

    def test_password_no_special_char(self):
        """Test password must have special character"""
        with pytest.raises(ValidationError) as exc_info:
            BrandCreate(
                name="Test Brand",
                email="test@example.com",
                password="NoSpecialChar123"
            )
        assert "special character" in str(exc_info.value)

    def test_invalid_email(self):
        """Test email validation"""
        with pytest.raises(ValidationError):
            BrandCreate(
                name="Test Brand",
                email="not-an-email",
                password="ValidPass123!"
            )


class TestSizeChartSchemas:
    """Tests for size chart schemas"""

    def test_size_chart_valid(self):
        """Test valid size chart creation"""
        chart = SizeChartCreate(
            size_name="M",
            chest_min=91,
            chest_max=99,
            waist_min=76,
            waist_max=84,
            fit_type="regular"
        )
        assert chart.size_name == "M"
        assert chart.chest_min == 91
        assert chart.fit_type == "regular"

    def test_size_chart_min_greater_than_max(self):
        """Test that min cannot be greater than max"""
        with pytest.raises(ValidationError) as exc_info:
            SizeChartCreate(
                size_name="M",
                chest_min=100,  # Greater than max
                chest_max=90,
                fit_type="regular"
            )
        assert "Chest" in str(exc_info.value) and "minimum" in str(exc_info.value)

    def test_size_chart_multiple_invalid_ranges(self):
        """Test multiple invalid ranges are all reported"""
        with pytest.raises(ValidationError) as exc_info:
            SizeChartCreate(
                size_name="M",
                chest_min=100,
                chest_max=90,
                waist_min=90,
                waist_max=80,
                fit_type="regular"
            )
        error_str = str(exc_info.value)
        assert "Chest" in error_str
        assert "Waist" in error_str

    def test_size_chart_invalid_fit_type(self):
        """Test invalid fit type is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            SizeChartCreate(
                size_name="M",
                chest_min=91,
                chest_max=99,
                fit_type="invalid_fit"
            )
        assert "fit_type" in str(exc_info.value)

    def test_size_chart_valid_fit_types(self):
        """Test all valid fit types are accepted"""
        for fit_type in ["tight", "regular", "loose"]:
            chart = SizeChartCreate(
                size_name="M",
                chest_min=91,
                chest_max=99,
                fit_type=fit_type
            )
            assert chart.fit_type == fit_type

    def test_size_chart_measurement_bounds(self):
        """Test measurement values are within bounds"""
        # Chest max is 300
        with pytest.raises(ValidationError):
            SizeChartCreate(
                size_name="M",
                chest_min=0,
                chest_max=350,  # Exceeds 300
                fit_type="regular"
            )

    def test_size_chart_partial_ranges(self):
        """Test that partial ranges (only min or only max) are allowed"""
        chart = SizeChartCreate(
            size_name="M",
            chest_min=91,
            # No chest_max
            waist_max=84,
            # No waist_min
            fit_type="regular"
        )
        assert chart.chest_min == 91
        assert chart.chest_max is None
        assert chart.waist_min is None
        assert chart.waist_max == 84


class TestProductSchemas:
    """Tests for product schemas"""

    def test_product_create_valid(self):
        """Test valid product creation"""
        product = ProductCreate(
            name="Test Shirt",
            category="tops",
            subcategory="t-shirt",
            gender="male",
            age_group="adult"
        )
        assert product.name == "Test Shirt"
        assert product.category == "tops"

    def test_product_name_required(self):
        """Test product name is required"""
        with pytest.raises(ValidationError):
            ProductCreate(
                category="tops"
            )

    def test_product_category_required(self):
        """Test product category is required"""
        with pytest.raises(ValidationError):
            ProductCreate(
                name="Test Shirt"
            )

    def test_product_with_size_chart(self):
        """Test product with size chart data"""
        product = ProductCreate(
            name="Test Shirt",
            category="tops",
            size_chart={"S": {"chest": 90}, "M": {"chest": 100}}
        )
        assert product.size_chart is not None
