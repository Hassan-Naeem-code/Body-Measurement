"""
Integration Tests for Product API Endpoints
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProductTemplatesAPI:
    """Tests for product template endpoints"""

    def test_list_templates(self, client):
        """Test listing available templates"""
        response = client.get("/api/v1/templates")
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert "total" in data
        assert data["total"] >= 7  # We have at least 7 templates

    def test_template_structure(self, client):
        """Test that template list has expected structure"""
        response = client.get("/api/v1/templates")
        assert response.status_code == 200
        templates = response.json()["templates"]

        for template in templates:
            assert "template_id" in template
            assert "name" in template
            assert "category" in template
            assert "size_count" in template
            assert "sizes" in template
            assert template["size_count"] > 0

    def test_size_chart_suggestions(self, client):
        """Test size chart suggestions endpoint"""
        response = client.get("/api/v1/size-chart-suggestions?category=tops&gender=male")
        assert response.status_code == 200
        data = response.json()
        assert "size_charts" in data
        assert data["category"] == "tops"
        assert data["gender"] == "male"
        assert len(data["size_charts"]) > 0


class TestProductCRUD:
    """Tests for product CRUD operations"""

    def test_create_product(self, client, auth_headers):
        """Test creating a product"""
        product_data = {
            "name": "Test T-Shirt",
            "category": "tops",
            "subcategory": "t-shirt",
            "gender": "male",
            "age_group": "adult"
        }
        response = client.post("/api/v1/products", json=product_data, headers=auth_headers)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test T-Shirt"
        assert data["category"] == "tops"
        assert "id" in data

    def test_create_product_unauthorized(self, client):
        """Test creating a product without auth fails"""
        product_data = {"name": "Test", "category": "tops"}
        response = client.post("/api/v1/products", json=product_data)
        assert response.status_code == 401

    def test_list_products(self, client, auth_headers, test_product):
        """Test listing products"""
        response = client.get("/api/v1/products", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "products" in data
        assert "total" in data
        assert data["total"] >= 1

    def test_get_product(self, client, auth_headers, test_product):
        """Test getting a specific product"""
        response = client.get(f"/api/v1/products/{test_product.id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(test_product.id)
        assert data["name"] == test_product.name

    def test_get_product_with_size_charts(self, client, auth_headers, test_product):
        """Test getting a product includes size charts"""
        response = client.get(f"/api/v1/products/{test_product.id}", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "size_charts" in data
        assert len(data["size_charts"]) == 3  # S, M, L

    def test_update_product(self, client, auth_headers, test_product):
        """Test updating a product"""
        update_data = {"name": "Updated T-Shirt", "description": "New description"}
        response = client.put(
            f"/api/v1/products/{test_product.id}",
            json=update_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated T-Shirt"
        assert data["description"] == "New description"

    def test_delete_product(self, client, auth_headers, test_product):
        """Test deleting a product"""
        response = client.delete(f"/api/v1/products/{test_product.id}", headers=auth_headers)
        assert response.status_code == 204

        # Verify it's deleted
        response = client.get(f"/api/v1/products/{test_product.id}", headers=auth_headers)
        assert response.status_code == 404


class TestProductFromTemplate:
    """Tests for creating products from templates"""

    def test_create_from_template(self, client, auth_headers):
        """Test creating a product from template"""
        response = client.post(
            "/api/v1/products/from-template?template_id=mens_tshirt",
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Men's T-Shirt"
        assert data["category"] == "tops"
        assert "size_charts" in data
        assert len(data["size_charts"]) == 7  # XS to 3XL

    def test_create_from_template_custom_name(self, client, auth_headers):
        """Test creating from template with custom name"""
        response = client.post(
            "/api/v1/products/from-template?template_id=mens_tshirt&name=My%20Custom%20Tee",
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "My Custom Tee"

    def test_create_from_invalid_template(self, client, auth_headers):
        """Test creating from invalid template fails"""
        response = client.post(
            "/api/v1/products/from-template?template_id=nonexistent",
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    def test_apply_template_to_product(self, client, auth_headers, test_product):
        """Test applying template to existing product"""
        response = client.post(
            f"/api/v1/products/{test_product.id}/apply-template?template_id=athletic_wear&replace_existing=true",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["size_charts"]) == 6  # Athletic has 6 sizes


class TestSizeChartEndpoints:
    """Tests for size chart CRUD endpoints"""

    def test_get_size_charts(self, client, auth_headers, test_product):
        """Test getting size charts for a product"""
        response = client.get(
            f"/api/v1/products/{test_product.id}/size-charts",
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_add_size_chart(self, client, auth_headers, test_product):
        """Test adding a size chart to a product"""
        chart_data = {
            "size_name": "XL",
            "chest_min": 107,
            "chest_max": 117,
            "waist_min": 94,
            "waist_max": 104,
            "fit_type": "regular"
        }
        response = client.post(
            f"/api/v1/products/{test_product.id}/size-charts",
            json=chart_data,
            headers=auth_headers
        )
        assert response.status_code == 201
        data = response.json()
        assert data["size_name"] == "XL"

    def test_add_duplicate_size_chart(self, client, auth_headers, test_product):
        """Test adding duplicate size chart fails"""
        chart_data = {
            "size_name": "M",  # Already exists
            "chest_min": 91,
            "chest_max": 99,
            "fit_type": "regular"
        }
        response = client.post(
            f"/api/v1/products/{test_product.id}/size-charts",
            json=chart_data,
            headers=auth_headers
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]


class TestSizeRecommendation:
    """Tests for size recommendation endpoints"""

    def test_recommend_size(self, client, auth_headers, test_product):
        """Test getting size recommendation"""
        request_data = {
            "product_id": str(test_product.id),
            "chest_circumference": 95,
            "waist_circumference": 80,
            "fit_preference": "regular"
        }
        response = client.post(
            f"/api/v1/products/{test_product.id}/recommend-size",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommended_size" in data
        assert "confidence" in data
        assert "fit_quality" in data
        assert data["recommended_size"] == "M"  # 95cm chest fits M

    def test_recommend_size_small(self, client, auth_headers, test_product):
        """Test size recommendation for small measurements"""
        request_data = {
            "product_id": str(test_product.id),
            "chest_circumference": 88,
            "waist_circumference": 73,
            "fit_preference": "regular"
        }
        response = client.post(
            f"/api/v1/products/{test_product.id}/recommend-size",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["recommended_size"] == "S"

    def test_recommend_size_large(self, client, auth_headers, test_product):
        """Test size recommendation for large measurements"""
        request_data = {
            "product_id": str(test_product.id),
            "chest_circumference": 102,
            "waist_circumference": 90,
            "fit_preference": "regular"
        }
        response = client.post(
            f"/api/v1/products/{test_product.id}/recommend-size",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert data["recommended_size"] == "L"

    def test_recommend_size_bulk(self, client, auth_headers, test_product):
        """Test bulk size recommendation"""
        request_data = {
            "chest_circumference": 95,
            "waist_circumference": 80,
            "fit_preference": "regular"
        }
        response = client.post(
            "/api/v1/recommend-size-bulk",
            json=request_data,
            headers=auth_headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert "total" in data
