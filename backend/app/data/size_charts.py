"""
Comprehensive Size Chart Data for Common Clothing Categories

This module provides pre-defined size charts based on industry standards
and major brand measurements. These can be used to seed the database
or as reference data for size recommendations.

Sources:
- ASTM International size standards
- Major US/EU retailer size charts
- ISO 3636 clothing size standards
"""

from typing import Dict, List, Any

# Standard US Men's Size Charts
MENS_TOPS_SIZES: List[Dict[str, Any]] = [
    {
        "size_name": "XS",
        "chest_min": 81, "chest_max": 86,
        "waist_min": 66, "waist_max": 71,
        "hip_min": 81, "hip_max": 86,
        "shoulder_width_min": 40, "shoulder_width_max": 42,
        "height_min": 160, "height_max": 170,
        "fit_type": "regular",
        "display_order": 1
    },
    {
        "size_name": "S",
        "chest_min": 86, "chest_max": 91,
        "waist_min": 71, "waist_max": 76,
        "hip_min": 86, "hip_max": 91,
        "shoulder_width_min": 42, "shoulder_width_max": 44,
        "height_min": 165, "height_max": 175,
        "fit_type": "regular",
        "display_order": 2
    },
    {
        "size_name": "M",
        "chest_min": 91, "chest_max": 99,
        "waist_min": 76, "waist_max": 84,
        "hip_min": 91, "hip_max": 99,
        "shoulder_width_min": 44, "shoulder_width_max": 46,
        "height_min": 170, "height_max": 180,
        "fit_type": "regular",
        "display_order": 3
    },
    {
        "size_name": "L",
        "chest_min": 99, "chest_max": 107,
        "waist_min": 84, "waist_max": 94,
        "hip_min": 99, "hip_max": 107,
        "shoulder_width_min": 46, "shoulder_width_max": 49,
        "height_min": 175, "height_max": 185,
        "fit_type": "regular",
        "display_order": 4
    },
    {
        "size_name": "XL",
        "chest_min": 107, "chest_max": 117,
        "waist_min": 94, "waist_max": 104,
        "hip_min": 107, "hip_max": 117,
        "shoulder_width_min": 49, "shoulder_width_max": 52,
        "height_min": 178, "height_max": 188,
        "fit_type": "regular",
        "display_order": 5
    },
    {
        "size_name": "2XL",
        "chest_min": 117, "chest_max": 127,
        "waist_min": 104, "waist_max": 114,
        "hip_min": 117, "hip_max": 127,
        "shoulder_width_min": 52, "shoulder_width_max": 55,
        "height_min": 180, "height_max": 190,
        "fit_type": "regular",
        "display_order": 6
    },
    {
        "size_name": "3XL",
        "chest_min": 127, "chest_max": 137,
        "waist_min": 114, "waist_max": 127,
        "hip_min": 127, "hip_max": 137,
        "shoulder_width_min": 55, "shoulder_width_max": 58,
        "height_min": 180, "height_max": 195,
        "fit_type": "regular",
        "display_order": 7
    }
]

# Standard US Women's Size Charts
WOMENS_TOPS_SIZES: List[Dict[str, Any]] = [
    {
        "size_name": "XS",
        "chest_min": 76, "chest_max": 81,
        "waist_min": 58, "waist_max": 63,
        "hip_min": 84, "hip_max": 89,
        "shoulder_width_min": 36, "shoulder_width_max": 38,
        "height_min": 155, "height_max": 165,
        "fit_type": "regular",
        "display_order": 1
    },
    {
        "size_name": "S",
        "chest_min": 81, "chest_max": 86,
        "waist_min": 63, "waist_max": 68,
        "hip_min": 89, "hip_max": 94,
        "shoulder_width_min": 38, "shoulder_width_max": 40,
        "height_min": 158, "height_max": 168,
        "fit_type": "regular",
        "display_order": 2
    },
    {
        "size_name": "M",
        "chest_min": 86, "chest_max": 94,
        "waist_min": 68, "waist_max": 76,
        "hip_min": 94, "hip_max": 102,
        "shoulder_width_min": 40, "shoulder_width_max": 42,
        "height_min": 160, "height_max": 172,
        "fit_type": "regular",
        "display_order": 3
    },
    {
        "size_name": "L",
        "chest_min": 94, "chest_max": 102,
        "waist_min": 76, "waist_max": 86,
        "hip_min": 102, "hip_max": 110,
        "shoulder_width_min": 42, "shoulder_width_max": 44,
        "height_min": 163, "height_max": 175,
        "fit_type": "regular",
        "display_order": 4
    },
    {
        "size_name": "XL",
        "chest_min": 102, "chest_max": 112,
        "waist_min": 86, "waist_max": 96,
        "hip_min": 110, "hip_max": 120,
        "shoulder_width_min": 44, "shoulder_width_max": 46,
        "height_min": 165, "height_max": 178,
        "fit_type": "regular",
        "display_order": 5
    },
    {
        "size_name": "2XL",
        "chest_min": 112, "chest_max": 122,
        "waist_min": 96, "waist_max": 108,
        "hip_min": 120, "hip_max": 130,
        "shoulder_width_min": 46, "shoulder_width_max": 48,
        "height_min": 165, "height_max": 180,
        "fit_type": "regular",
        "display_order": 6
    }
]

# Men's Pants/Bottoms
MENS_BOTTOMS_SIZES: List[Dict[str, Any]] = [
    {
        "size_name": "28",
        "waist_min": 71, "waist_max": 74,
        "hip_min": 86, "hip_max": 89,
        "inseam_min": 76, "inseam_max": 81,
        "height_min": 165, "height_max": 175,
        "fit_type": "regular",
        "display_order": 1
    },
    {
        "size_name": "30",
        "waist_min": 76, "waist_max": 79,
        "hip_min": 91, "hip_max": 94,
        "inseam_min": 76, "inseam_max": 81,
        "height_min": 168, "height_max": 178,
        "fit_type": "regular",
        "display_order": 2
    },
    {
        "size_name": "32",
        "waist_min": 81, "waist_max": 84,
        "hip_min": 97, "hip_max": 99,
        "inseam_min": 79, "inseam_max": 84,
        "height_min": 170, "height_max": 182,
        "fit_type": "regular",
        "display_order": 3
    },
    {
        "size_name": "34",
        "waist_min": 86, "waist_max": 89,
        "hip_min": 102, "hip_max": 104,
        "inseam_min": 79, "inseam_max": 84,
        "height_min": 173, "height_max": 185,
        "fit_type": "regular",
        "display_order": 4
    },
    {
        "size_name": "36",
        "waist_min": 91, "waist_max": 94,
        "hip_min": 107, "hip_max": 110,
        "inseam_min": 81, "inseam_max": 86,
        "height_min": 175, "height_max": 188,
        "fit_type": "regular",
        "display_order": 5
    },
    {
        "size_name": "38",
        "waist_min": 97, "waist_max": 102,
        "hip_min": 112, "hip_max": 117,
        "inseam_min": 81, "inseam_max": 86,
        "height_min": 178, "height_max": 190,
        "fit_type": "regular",
        "display_order": 6
    },
    {
        "size_name": "40",
        "waist_min": 102, "waist_max": 107,
        "hip_min": 117, "hip_max": 122,
        "inseam_min": 81, "inseam_max": 86,
        "height_min": 178, "height_max": 193,
        "fit_type": "regular",
        "display_order": 7
    }
]

# Women's Pants/Bottoms (US Numeric)
WOMENS_BOTTOMS_SIZES: List[Dict[str, Any]] = [
    {
        "size_name": "0/XS",
        "waist_min": 58, "waist_max": 63,
        "hip_min": 84, "hip_max": 89,
        "inseam_min": 71, "inseam_max": 76,
        "height_min": 155, "height_max": 165,
        "fit_type": "regular",
        "display_order": 1
    },
    {
        "size_name": "2/S",
        "waist_min": 63, "waist_max": 66,
        "hip_min": 89, "hip_max": 91,
        "inseam_min": 74, "inseam_max": 79,
        "height_min": 158, "height_max": 168,
        "fit_type": "regular",
        "display_order": 2
    },
    {
        "size_name": "4/S",
        "waist_min": 66, "waist_max": 69,
        "hip_min": 91, "hip_max": 94,
        "inseam_min": 74, "inseam_max": 79,
        "height_min": 158, "height_max": 170,
        "fit_type": "regular",
        "display_order": 3
    },
    {
        "size_name": "6/M",
        "waist_min": 69, "waist_max": 71,
        "hip_min": 94, "hip_max": 97,
        "inseam_min": 76, "inseam_max": 81,
        "height_min": 160, "height_max": 172,
        "fit_type": "regular",
        "display_order": 4
    },
    {
        "size_name": "8/M",
        "waist_min": 71, "waist_max": 76,
        "hip_min": 97, "hip_max": 102,
        "inseam_min": 76, "inseam_max": 81,
        "height_min": 160, "height_max": 175,
        "fit_type": "regular",
        "display_order": 5
    },
    {
        "size_name": "10/L",
        "waist_min": 76, "waist_max": 81,
        "hip_min": 102, "hip_max": 107,
        "inseam_min": 79, "inseam_max": 84,
        "height_min": 163, "height_max": 175,
        "fit_type": "regular",
        "display_order": 6
    },
    {
        "size_name": "12/L",
        "waist_min": 81, "waist_max": 86,
        "hip_min": 107, "hip_max": 112,
        "inseam_min": 79, "inseam_max": 84,
        "height_min": 163, "height_max": 178,
        "fit_type": "regular",
        "display_order": 7
    },
    {
        "size_name": "14/XL",
        "waist_min": 86, "waist_max": 91,
        "hip_min": 112, "hip_max": 117,
        "inseam_min": 79, "inseam_max": 84,
        "height_min": 165, "height_max": 178,
        "fit_type": "regular",
        "display_order": 8
    }
]

# Dress Sizes (US)
WOMENS_DRESS_SIZES: List[Dict[str, Any]] = [
    {
        "size_name": "XS (0-2)",
        "chest_min": 76, "chest_max": 84,
        "waist_min": 58, "waist_max": 66,
        "hip_min": 84, "hip_max": 91,
        "height_min": 155, "height_max": 168,
        "fit_type": "regular",
        "display_order": 1
    },
    {
        "size_name": "S (4-6)",
        "chest_min": 84, "chest_max": 89,
        "waist_min": 66, "waist_max": 71,
        "hip_min": 91, "hip_max": 97,
        "height_min": 158, "height_max": 170,
        "fit_type": "regular",
        "display_order": 2
    },
    {
        "size_name": "M (8-10)",
        "chest_min": 89, "chest_max": 97,
        "waist_min": 71, "waist_max": 81,
        "hip_min": 97, "hip_max": 107,
        "height_min": 160, "height_max": 175,
        "fit_type": "regular",
        "display_order": 3
    },
    {
        "size_name": "L (12-14)",
        "chest_min": 97, "chest_max": 107,
        "waist_min": 81, "waist_max": 91,
        "hip_min": 107, "hip_max": 117,
        "height_min": 163, "height_max": 178,
        "fit_type": "regular",
        "display_order": 4
    },
    {
        "size_name": "XL (16-18)",
        "chest_min": 107, "chest_max": 117,
        "waist_min": 91, "waist_max": 102,
        "hip_min": 117, "hip_max": 127,
        "height_min": 165, "height_max": 180,
        "fit_type": "regular",
        "display_order": 5
    }
]

# Athletic/Sports Sizes (Unisex)
ATHLETIC_SIZES: List[Dict[str, Any]] = [
    {
        "size_name": "XS",
        "chest_min": 78, "chest_max": 84,
        "waist_min": 61, "waist_max": 68,
        "hip_min": 81, "hip_max": 86,
        "height_min": 155, "height_max": 165,
        "fit_type": "tight",  # Athletic fit is typically tighter
        "display_order": 1
    },
    {
        "size_name": "S",
        "chest_min": 84, "chest_max": 91,
        "waist_min": 68, "waist_max": 74,
        "hip_min": 86, "hip_max": 94,
        "height_min": 163, "height_max": 173,
        "fit_type": "tight",
        "display_order": 2
    },
    {
        "size_name": "M",
        "chest_min": 91, "chest_max": 99,
        "waist_min": 74, "waist_max": 81,
        "hip_min": 94, "hip_max": 102,
        "height_min": 168, "height_max": 178,
        "fit_type": "tight",
        "display_order": 3
    },
    {
        "size_name": "L",
        "chest_min": 99, "chest_max": 109,
        "waist_min": 81, "waist_max": 91,
        "hip_min": 102, "hip_max": 112,
        "height_min": 175, "height_max": 185,
        "fit_type": "tight",
        "display_order": 4
    },
    {
        "size_name": "XL",
        "chest_min": 109, "chest_max": 119,
        "waist_min": 91, "waist_max": 102,
        "hip_min": 112, "hip_max": 122,
        "height_min": 178, "height_max": 190,
        "fit_type": "tight",
        "display_order": 5
    },
    {
        "size_name": "2XL",
        "chest_min": 119, "chest_max": 130,
        "waist_min": 102, "waist_max": 114,
        "hip_min": 122, "hip_max": 132,
        "height_min": 180, "height_max": 195,
        "fit_type": "tight",
        "display_order": 6
    }
]

# Outerwear/Jacket Sizes (Men's)
MENS_OUTERWEAR_SIZES: List[Dict[str, Any]] = [
    {
        "size_name": "S",
        "chest_min": 89, "chest_max": 97,
        "waist_min": 74, "waist_max": 81,
        "shoulder_width_min": 43, "shoulder_width_max": 45,
        "arm_length_min": 61, "arm_length_max": 64,
        "height_min": 165, "height_max": 175,
        "fit_type": "regular",
        "display_order": 1
    },
    {
        "size_name": "M",
        "chest_min": 97, "chest_max": 104,
        "waist_min": 81, "waist_max": 89,
        "shoulder_width_min": 45, "shoulder_width_max": 47,
        "arm_length_min": 64, "arm_length_max": 66,
        "height_min": 170, "height_max": 180,
        "fit_type": "regular",
        "display_order": 2
    },
    {
        "size_name": "L",
        "chest_min": 104, "chest_max": 112,
        "waist_min": 89, "waist_max": 97,
        "shoulder_width_min": 47, "shoulder_width_max": 50,
        "arm_length_min": 66, "arm_length_max": 69,
        "height_min": 175, "height_max": 185,
        "fit_type": "regular",
        "display_order": 3
    },
    {
        "size_name": "XL",
        "chest_min": 112, "chest_max": 122,
        "waist_min": 97, "waist_max": 107,
        "shoulder_width_min": 50, "shoulder_width_max": 53,
        "arm_length_min": 69, "arm_length_max": 71,
        "height_min": 178, "height_max": 190,
        "fit_type": "regular",
        "display_order": 4
    },
    {
        "size_name": "2XL",
        "chest_min": 122, "chest_max": 132,
        "waist_min": 107, "waist_max": 117,
        "shoulder_width_min": 53, "shoulder_width_max": 56,
        "arm_length_min": 71, "arm_length_max": 74,
        "height_min": 180, "height_max": 195,
        "fit_type": "regular",
        "display_order": 5
    }
]

# EU Size Conversions (for reference)
EU_TO_US_MENS = {
    "44": "XS",
    "46": "S",
    "48": "M",
    "50": "L",
    "52": "XL",
    "54": "2XL",
    "56": "3XL"
}

EU_TO_US_WOMENS = {
    "32": "XS (0)",
    "34": "S (2-4)",
    "36": "S (4-6)",
    "38": "M (6-8)",
    "40": "M (8-10)",
    "42": "L (10-12)",
    "44": "L (12-14)",
    "46": "XL (14-16)"
}

# UK Size Conversions
UK_TO_US_WOMENS = {
    "4": "0",
    "6": "2",
    "8": "4",
    "10": "6",
    "12": "8",
    "14": "10",
    "16": "12",
    "18": "14"
}

# Pre-defined product templates
PRODUCT_TEMPLATES = {
    "mens_tshirt": {
        "name": "Men's T-Shirt",
        "category": "tops",
        "subcategory": "t-shirt",
        "gender": "male",
        "age_group": "adult",
        "size_charts": MENS_TOPS_SIZES
    },
    "womens_blouse": {
        "name": "Women's Blouse",
        "category": "tops",
        "subcategory": "blouse",
        "gender": "female",
        "age_group": "adult",
        "size_charts": WOMENS_TOPS_SIZES
    },
    "mens_jeans": {
        "name": "Men's Jeans",
        "category": "bottoms",
        "subcategory": "jeans",
        "gender": "male",
        "age_group": "adult",
        "size_charts": MENS_BOTTOMS_SIZES
    },
    "womens_pants": {
        "name": "Women's Pants",
        "category": "bottoms",
        "subcategory": "pants",
        "gender": "female",
        "age_group": "adult",
        "size_charts": WOMENS_BOTTOMS_SIZES
    },
    "womens_dress": {
        "name": "Women's Dress",
        "category": "dresses",
        "subcategory": "dress",
        "gender": "female",
        "age_group": "adult",
        "size_charts": WOMENS_DRESS_SIZES
    },
    "athletic_wear": {
        "name": "Athletic Top",
        "category": "tops",
        "subcategory": "athletic",
        "gender": "unisex",
        "age_group": "adult",
        "size_charts": ATHLETIC_SIZES
    },
    "mens_jacket": {
        "name": "Men's Jacket",
        "category": "outerwear",
        "subcategory": "jacket",
        "gender": "male",
        "age_group": "adult",
        "size_charts": MENS_OUTERWEAR_SIZES
    }
}


def get_size_charts_for_category(category: str, gender: str = None) -> List[Dict[str, Any]]:
    """
    Get appropriate size charts for a category and gender

    Args:
        category: Product category (tops, bottoms, dresses, outerwear)
        gender: Gender (male, female, unisex) - optional

    Returns:
        List of size chart dictionaries
    """
    if category == "tops":
        if gender == "female":
            return WOMENS_TOPS_SIZES
        elif gender == "male":
            return MENS_TOPS_SIZES
        else:
            return ATHLETIC_SIZES  # Default to unisex athletic

    elif category == "bottoms":
        if gender == "female":
            return WOMENS_BOTTOMS_SIZES
        else:
            return MENS_BOTTOMS_SIZES

    elif category == "dresses":
        return WOMENS_DRESS_SIZES

    elif category == "outerwear":
        return MENS_OUTERWEAR_SIZES

    else:
        # Default to men's tops
        return MENS_TOPS_SIZES


def get_product_template(template_name: str) -> Dict[str, Any]:
    """
    Get a pre-defined product template with size charts

    Args:
        template_name: One of the template names (e.g., 'mens_tshirt')

    Returns:
        Product template dictionary with size charts
    """
    return PRODUCT_TEMPLATES.get(template_name, PRODUCT_TEMPLATES["mens_tshirt"])
