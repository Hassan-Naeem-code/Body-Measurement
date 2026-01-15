"""
Integration Tests for Authentication API Endpoints
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBrandRegistration:
    """Tests for brand registration"""

    def test_register_brand(self, client):
        """Test successful brand registration"""
        brand_data = {
            "name": "New Brand",
            "email": "newbrand@example.com",
            "password": "SecurePass123!"
        }
        response = client.post("/api/v1/brands/register", json=brand_data)
        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "brand" in data
        assert data["brand"]["name"] == "New Brand"
        assert data["brand"]["email"] == "newbrand@example.com"
        assert "api_key" in data["brand"]

    def test_register_duplicate_email(self, client, test_brand):
        """Test registration with existing email fails"""
        brand_data = {
            "name": "Another Brand",
            "email": test_brand.email,  # Same email as existing brand
            "password": "SecurePass123!"
        }
        response = client.post("/api/v1/brands/register", json=brand_data)
        assert response.status_code == 400
        assert "already registered" in response.json()["detail"]

    def test_register_weak_password(self, client):
        """Test registration with weak password fails"""
        brand_data = {
            "name": "New Brand",
            "email": "test@example.com",
            "password": "weak"
        }
        response = client.post("/api/v1/brands/register", json=brand_data)
        assert response.status_code == 422  # Validation error

    def test_register_invalid_email(self, client):
        """Test registration with invalid email fails"""
        brand_data = {
            "name": "New Brand",
            "email": "not-an-email",
            "password": "SecurePass123!"
        }
        response = client.post("/api/v1/brands/register", json=brand_data)
        assert response.status_code == 422  # Validation error


class TestBrandLogin:
    """Tests for brand login"""

    def test_login_success(self, client, test_brand):
        """Test successful login"""
        login_data = {
            "email": test_brand.email,
            "password": "TestPass123!"
        }
        response = client.post("/api/v1/brands/login", json=login_data)
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client, test_brand):
        """Test login with wrong password fails"""
        login_data = {
            "email": test_brand.email,
            "password": "WrongPassword123!"
        }
        response = client.post("/api/v1/brands/login", json=login_data)
        assert response.status_code == 401

    def test_login_nonexistent_email(self, client):
        """Test login with nonexistent email fails"""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "SomePass123!"
        }
        response = client.post("/api/v1/brands/login", json=login_data)
        assert response.status_code == 401


class TestAPIKeyAuth:
    """Tests for API key authentication"""

    def test_auth_with_header(self, client, test_brand):
        """Test authentication with X-API-Key header"""
        response = client.get(
            "/api/v1/brands/me",
            headers={"X-API-Key": test_brand.api_key}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_brand.id)

    def test_auth_with_query_param_deprecated(self, client, test_brand):
        """Test authentication with deprecated query parameter still works"""
        response = client.get(f"/api/v1/brands/me?api_key={test_brand.api_key}")
        assert response.status_code == 200

    def test_auth_with_invalid_api_key(self, client):
        """Test authentication with invalid API key fails"""
        response = client.get(
            "/api/v1/brands/me",
            headers={"X-API-Key": "invalid-api-key"}
        )
        assert response.status_code == 401

    def test_auth_without_api_key(self, client):
        """Test authentication without API key fails"""
        response = client.get("/api/v1/brands/me")
        assert response.status_code == 401
        assert "API key required" in response.json()["detail"]


class TestBrandProfile:
    """Tests for brand profile endpoints"""

    def test_get_profile(self, client, auth_headers, test_brand):
        """Test getting brand profile"""
        response = client.get("/api/v1/brands/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == test_brand.name
        assert data["email"] == test_brand.email
        assert data["is_active"] == True

    def test_get_usage_stats(self, client, auth_headers):
        """Test getting usage statistics"""
        response = client.get("/api/v1/brands/usage", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "requests_today" in data
        assert "plan_limit" in data

    def test_regenerate_api_key(self, client, auth_headers, test_brand):
        """Test regenerating API key"""
        old_key = test_brand.api_key
        response = client.post("/api/v1/brands/regenerate-key", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "api_key" in data
        assert data["api_key"] != old_key
