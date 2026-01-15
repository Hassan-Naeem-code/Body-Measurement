"""
Unit Tests for Size Chart Data and Templates
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.size_charts import (
    MENS_TOPS_SIZES,
    WOMENS_TOPS_SIZES,
    MENS_BOTTOMS_SIZES,
    WOMENS_BOTTOMS_SIZES,
    WOMENS_DRESS_SIZES,
    ATHLETIC_SIZES,
    MENS_OUTERWEAR_SIZES,
    PRODUCT_TEMPLATES,
    get_size_charts_for_category,
    get_product_template,
)


class TestSizeChartData:
    """Tests for size chart data integrity"""

    @pytest.mark.parametrize("sizes,name", [
        (MENS_TOPS_SIZES, "Men's Tops"),
        (WOMENS_TOPS_SIZES, "Women's Tops"),
        (MENS_BOTTOMS_SIZES, "Men's Bottoms"),
        (WOMENS_BOTTOMS_SIZES, "Women's Bottoms"),
        (WOMENS_DRESS_SIZES, "Women's Dresses"),
        (ATHLETIC_SIZES, "Athletic"),
        (MENS_OUTERWEAR_SIZES, "Men's Outerwear"),
    ])
    def test_size_chart_not_empty(self, sizes, name):
        """Test that each size chart has at least one size"""
        assert len(sizes) > 0, f"{name} size chart should not be empty"

    @pytest.mark.parametrize("sizes,name", [
        (MENS_TOPS_SIZES, "Men's Tops"),
        (WOMENS_TOPS_SIZES, "Women's Tops"),
        (MENS_BOTTOMS_SIZES, "Men's Bottoms"),
        (WOMENS_BOTTOMS_SIZES, "Women's Bottoms"),
        (WOMENS_DRESS_SIZES, "Women's Dresses"),
        (ATHLETIC_SIZES, "Athletic"),
        (MENS_OUTERWEAR_SIZES, "Men's Outerwear"),
    ])
    def test_size_chart_has_required_fields(self, sizes, name):
        """Test that each size has required fields"""
        for size in sizes:
            assert "size_name" in size, f"{name} size missing size_name"
            assert "fit_type" in size, f"{name} size missing fit_type"
            assert "display_order" in size, f"{name} size missing display_order"

    @pytest.mark.parametrize("sizes,name", [
        (MENS_TOPS_SIZES, "Men's Tops"),
        (WOMENS_TOPS_SIZES, "Women's Tops"),
        (MENS_BOTTOMS_SIZES, "Men's Bottoms"),
        (WOMENS_BOTTOMS_SIZES, "Women's Bottoms"),
    ])
    def test_size_chart_min_max_valid(self, sizes, name):
        """Test that min values are less than max values"""
        measurement_pairs = [
            ("chest_min", "chest_max"),
            ("waist_min", "waist_max"),
            ("hip_min", "hip_max"),
            ("height_min", "height_max"),
            ("inseam_min", "inseam_max"),
        ]

        for size in sizes:
            for min_key, max_key in measurement_pairs:
                min_val = size.get(min_key)
                max_val = size.get(max_key)
                if min_val is not None and max_val is not None:
                    assert min_val <= max_val, (
                        f"{name} {size['size_name']}: {min_key}={min_val} > {max_key}={max_val}"
                    )

    def test_sizes_are_ordered(self):
        """Test that sizes are properly ordered by display_order"""
        for sizes in [MENS_TOPS_SIZES, WOMENS_TOPS_SIZES, ATHLETIC_SIZES]:
            orders = [s["display_order"] for s in sizes]
            assert orders == sorted(orders), "Sizes should be in display order"

    def test_unique_size_names(self):
        """Test that size names are unique within each chart"""
        for sizes in [MENS_TOPS_SIZES, WOMENS_TOPS_SIZES, MENS_BOTTOMS_SIZES]:
            names = [s["size_name"] for s in sizes]
            assert len(names) == len(set(names)), "Size names should be unique"

    def test_measurement_values_realistic(self):
        """Test that measurement values are within realistic ranges"""
        for sizes in [MENS_TOPS_SIZES, WOMENS_TOPS_SIZES]:
            for size in sizes:
                # Chest should be between 70-150 cm for most people
                if "chest_min" in size and size["chest_min"] is not None:
                    assert 70 <= size["chest_min"] <= 150
                if "chest_max" in size and size["chest_max"] is not None:
                    assert 70 <= size["chest_max"] <= 150

                # Waist should be between 50-130 cm
                if "waist_min" in size and size["waist_min"] is not None:
                    assert 50 <= size["waist_min"] <= 130
                if "waist_max" in size and size["waist_max"] is not None:
                    assert 50 <= size["waist_max"] <= 130


class TestProductTemplates:
    """Tests for product templates"""

    def test_all_templates_exist(self):
        """Test that all expected templates exist"""
        expected_templates = [
            "mens_tshirt",
            "womens_blouse",
            "mens_jeans",
            "womens_pants",
            "womens_dress",
            "athletic_wear",
            "mens_jacket",
        ]
        for template_id in expected_templates:
            assert template_id in PRODUCT_TEMPLATES, f"Missing template: {template_id}"

    def test_template_structure(self):
        """Test that templates have required fields"""
        required_fields = ["name", "category", "size_charts"]
        for template_id, template in PRODUCT_TEMPLATES.items():
            for field in required_fields:
                assert field in template, f"Template {template_id} missing field: {field}"

    def test_template_size_charts_not_empty(self):
        """Test that templates have size charts"""
        for template_id, template in PRODUCT_TEMPLATES.items():
            assert len(template["size_charts"]) > 0, f"Template {template_id} has no size charts"

    def test_get_product_template(self):
        """Test get_product_template function"""
        template = get_product_template("mens_tshirt")
        assert template["name"] == "Men's T-Shirt"
        assert template["category"] == "tops"
        assert len(template["size_charts"]) > 0

    def test_get_product_template_default(self):
        """Test that invalid template returns default"""
        template = get_product_template("nonexistent")
        # Should return default (mens_tshirt)
        assert template is not None
        assert template["name"] == "Men's T-Shirt"


class TestGetSizeChartsForCategory:
    """Tests for category-based size chart lookup"""

    def test_tops_male(self):
        """Test getting men's tops size charts"""
        charts = get_size_charts_for_category("tops", "male")
        assert charts == MENS_TOPS_SIZES

    def test_tops_female(self):
        """Test getting women's tops size charts"""
        charts = get_size_charts_for_category("tops", "female")
        assert charts == WOMENS_TOPS_SIZES

    def test_tops_unisex(self):
        """Test getting unisex (athletic) size charts"""
        charts = get_size_charts_for_category("tops", "unisex")
        assert charts == ATHLETIC_SIZES

    def test_tops_no_gender(self):
        """Test getting size charts when no gender specified"""
        charts = get_size_charts_for_category("tops", None)
        assert charts == ATHLETIC_SIZES

    def test_bottoms_male(self):
        """Test getting men's bottoms"""
        charts = get_size_charts_for_category("bottoms", "male")
        assert charts == MENS_BOTTOMS_SIZES

    def test_bottoms_female(self):
        """Test getting women's bottoms"""
        charts = get_size_charts_for_category("bottoms", "female")
        assert charts == WOMENS_BOTTOMS_SIZES

    def test_dresses(self):
        """Test getting dress sizes (always women's)"""
        charts = get_size_charts_for_category("dresses", None)
        assert charts == WOMENS_DRESS_SIZES

    def test_outerwear(self):
        """Test getting outerwear sizes"""
        charts = get_size_charts_for_category("outerwear", None)
        assert charts == MENS_OUTERWEAR_SIZES

    def test_unknown_category(self):
        """Test that unknown category returns default"""
        charts = get_size_charts_for_category("unknown", None)
        assert charts == MENS_TOPS_SIZES  # Default


class TestSizeChartCoverage:
    """Tests for size chart coverage and gaps"""

    def test_no_gaps_in_chest_ranges(self):
        """Test that chest ranges don't have gaps"""
        for name, sizes in [("Men's Tops", MENS_TOPS_SIZES), ("Women's Tops", WOMENS_TOPS_SIZES)]:
            sorted_sizes = sorted(sizes, key=lambda x: x.get("chest_min", 0) or 0)
            for i in range(len(sorted_sizes) - 1):
                current_max = sorted_sizes[i].get("chest_max", 0)
                next_min = sorted_sizes[i + 1].get("chest_min", 0)
                if current_max and next_min:
                    # Allow small gap (1-2 cm) or overlap
                    gap = next_min - current_max
                    assert gap <= 2, (
                        f"{name}: Gap of {gap}cm between {sorted_sizes[i]['size_name']} "
                        f"and {sorted_sizes[i+1]['size_name']}"
                    )

    def test_sizes_cover_common_range(self):
        """Test that sizes cover common measurement ranges"""
        # Common chest range for men: 85-115 cm
        mens_min = min(s.get("chest_min", 999) for s in MENS_TOPS_SIZES if s.get("chest_min"))
        mens_max = max(s.get("chest_max", 0) for s in MENS_TOPS_SIZES if s.get("chest_max"))
        assert mens_min <= 86, "Men's sizes should cover small chests"
        assert mens_max >= 115, "Men's sizes should cover large chests"

        # Common chest range for women: 75-110 cm
        womens_min = min(s.get("chest_min", 999) for s in WOMENS_TOPS_SIZES if s.get("chest_min"))
        womens_max = max(s.get("chest_max", 0) for s in WOMENS_TOPS_SIZES if s.get("chest_max"))
        assert womens_min <= 80, "Women's sizes should cover small chests"
        assert womens_max >= 110, "Women's sizes should cover large chests"
