"""
Product Management API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional
from uuid import UUID

from app.core.database import get_db
from app.core.auth import get_current_brand
from app.models.brand import Brand
from app.models.product import Product, SizeChart
from app.schemas.product import (
    ProductCreate,
    ProductUpdate,
    ProductResponse,
    ProductWithSizeCharts,
    ProductListResponse,
    SizeChartCreate,
    SizeChartUpdate,
    SizeChartResponse,
)

router = APIRouter()


# ===== Product CRUD Endpoints =====

@router.post("/products", response_model=ProductWithSizeCharts, status_code=status.HTTP_201_CREATED)
def create_product(
    product_data: ProductCreate,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Create a new product with optional size charts

    **Authentication**: Requires valid API key in X-API-Key header

    **Request Body**:
    - name: Product name (required)
    - category: Product category - tops, bottoms, dresses, outerwear (required)
    - subcategory: Product subcategory (optional)
    - gender: Target gender - male, female, unisex (optional)
    - age_group: Target age group - adult, teen, child (optional)
    - size_charts: List of size charts (optional - can add later)

    **Example**:
    ```json
    {
      "name": "Classic Fit T-Shirt",
      "sku": "TSH-001",
      "category": "tops",
      "subcategory": "t-shirt",
      "gender": "male",
      "age_group": "adult",
      "description": "Comfortable cotton t-shirt",
      "size_charts": [
        {
          "size_name": "M",
          "chest_min": 96,
          "chest_max": 102,
          "waist_min": 81,
          "waist_max": 87,
          "fit_type": "regular"
        }
      ]
    }
    ```
    """
    # Create product
    product = Product(
        brand_id=current_brand.id,
        name=product_data.name,
        sku=product_data.sku,
        category=product_data.category,
        subcategory=product_data.subcategory,
        gender=product_data.gender,
        age_group=product_data.age_group,
        description=product_data.description,
        image_url=product_data.image_url,
        is_active=product_data.is_active,
    )

    db.add(product)
    db.flush()  # Get product ID without committing

    # Add size charts if provided
    if product_data.size_charts:
        for chart_data in product_data.size_charts:
            # Check for duplicate size names
            existing = db.query(SizeChart).filter(
                and_(
                    SizeChart.product_id == product.id,
                    SizeChart.size_name == chart_data.size_name
                )
            ).first()

            if existing:
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Size '{chart_data.size_name}' already exists for this product"
                )

            size_chart = SizeChart(
                product_id=product.id,
                size_name=chart_data.size_name,
                chest_min=chart_data.chest_min,
                chest_max=chart_data.chest_max,
                waist_min=chart_data.waist_min,
                waist_max=chart_data.waist_max,
                hip_min=chart_data.hip_min,
                hip_max=chart_data.hip_max,
                height_min=chart_data.height_min,
                height_max=chart_data.height_max,
                inseam_min=chart_data.inseam_min,
                inseam_max=chart_data.inseam_max,
                shoulder_width_min=chart_data.shoulder_width_min,
                shoulder_width_max=chart_data.shoulder_width_max,
                arm_length_min=chart_data.arm_length_min,
                arm_length_max=chart_data.arm_length_max,
                weight_min=chart_data.weight_min,
                weight_max=chart_data.weight_max,
                fit_type=chart_data.fit_type,
                display_order=chart_data.display_order,
            )
            db.add(size_chart)

    db.commit()
    db.refresh(product)

    return product


@router.get("/products", response_model=ProductListResponse)
def list_products(
    skip: int = Query(0, ge=0, description="Number of products to skip"),
    limit: int = Query(100, ge=1, le=500, description="Number of products to return"),
    category: Optional[str] = Query(None, description="Filter by category"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    List all products for the authenticated brand

    **Authentication**: Requires valid API key in X-API-Key header

    **Query Parameters**:
    - skip: Number of products to skip (pagination)
    - limit: Number of products to return (max 500)
    - category: Filter by category (tops, bottoms, dresses, outerwear)
    - is_active: Filter by active status (true/false)
    """
    query = db.query(Product).filter(Product.brand_id == current_brand.id)

    # Apply filters
    if category:
        query = query.filter(Product.category == category)
    if is_active is not None:
        query = query.filter(Product.is_active == is_active)

    # Get total count
    total = query.count()

    # Apply pagination and get products
    products = query.order_by(Product.created_at.desc()).offset(skip).limit(limit).all()

    return ProductListResponse(total=total, products=products)


@router.get("/products/{product_id}", response_model=ProductWithSizeCharts)
def get_product(
    product_id: UUID,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Get a specific product with its size charts

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - product_id: UUID of the product
    """
    product = db.query(Product).filter(
        and_(
            Product.id == product_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )

    return product


@router.put("/products/{product_id}", response_model=ProductWithSizeCharts)
def update_product(
    product_id: UUID,
    product_data: ProductUpdate,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Update a product's metadata (not size charts - use separate endpoints for those)

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - product_id: UUID of the product

    **Request Body**: All fields optional
    - name: Product name
    - sku: Product SKU
    - category: Product category
    - subcategory: Product subcategory
    - gender: Target gender
    - age_group: Target age group
    - description: Product description
    - image_url: Product image URL
    - is_active: Active status
    """
    product = db.query(Product).filter(
        and_(
            Product.id == product_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )

    # Update fields (only if provided)
    update_data = product_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(product, field, value)

    db.commit()
    db.refresh(product)

    return product


@router.delete("/products/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_product(
    product_id: UUID,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Delete a product (and all its size charts due to CASCADE)

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - product_id: UUID of the product
    """
    product = db.query(Product).filter(
        and_(
            Product.id == product_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )

    db.delete(product)
    db.commit()

    return None


# ===== Size Chart Endpoints =====

@router.post("/products/{product_id}/size-charts", response_model=SizeChartResponse, status_code=status.HTTP_201_CREATED)
def add_size_chart(
    product_id: UUID,
    chart_data: SizeChartCreate,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Add a size chart to a product

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - product_id: UUID of the product

    **Request Body**:
    - size_name: Size name (XS, S, M, L, XL, etc.) - required
    - chest_min/max: Chest circumference range in cm
    - waist_min/max: Waist circumference range in cm
    - hip_min/max: Hip circumference range in cm
    - height_min/max: Height range in cm
    - fit_type: Fit type (tight, regular, loose) - default: regular

    **Example**:
    ```json
    {
      "size_name": "M",
      "chest_min": 96,
      "chest_max": 102,
      "waist_min": 81,
      "waist_max": 87,
      "hip_min": 96,
      "hip_max": 102,
      "height_min": 170,
      "height_max": 180,
      "fit_type": "regular"
    }
    ```
    """
    # Verify product exists and belongs to brand
    product = db.query(Product).filter(
        and_(
            Product.id == product_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )

    # Check for duplicate size name
    existing = db.query(SizeChart).filter(
        and_(
            SizeChart.product_id == product_id,
            SizeChart.size_name == chart_data.size_name
        )
    ).first()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Size '{chart_data.size_name}' already exists for this product. Use PUT to update."
        )

    # Create size chart
    size_chart = SizeChart(
        product_id=product_id,
        size_name=chart_data.size_name,
        chest_min=chart_data.chest_min,
        chest_max=chart_data.chest_max,
        waist_min=chart_data.waist_min,
        waist_max=chart_data.waist_max,
        hip_min=chart_data.hip_min,
        hip_max=chart_data.hip_max,
        height_min=chart_data.height_min,
        height_max=chart_data.height_max,
        inseam_min=chart_data.inseam_min,
        inseam_max=chart_data.inseam_max,
        shoulder_width_min=chart_data.shoulder_width_min,
        shoulder_width_max=chart_data.shoulder_width_max,
        arm_length_min=chart_data.arm_length_min,
        arm_length_max=chart_data.arm_length_max,
        weight_min=chart_data.weight_min,
        weight_max=chart_data.weight_max,
        fit_type=chart_data.fit_type,
        display_order=chart_data.display_order,
    )

    db.add(size_chart)
    db.commit()
    db.refresh(size_chart)

    return size_chart


@router.get("/products/{product_id}/size-charts", response_model=List[SizeChartResponse])
def get_size_charts(
    product_id: UUID,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Get all size charts for a product

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - product_id: UUID of the product
    """
    # Verify product exists and belongs to brand
    product = db.query(Product).filter(
        and_(
            Product.id == product_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )

    size_charts = db.query(SizeChart).filter(
        SizeChart.product_id == product_id
    ).order_by(SizeChart.display_order, SizeChart.size_name).all()

    return size_charts


@router.put("/size-charts/{chart_id}", response_model=SizeChartResponse)
def update_size_chart(
    chart_id: UUID,
    chart_data: SizeChartUpdate,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Update a size chart

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - chart_id: UUID of the size chart

    **Request Body**: All fields optional
    - size_name: Size name
    - chest_min/max: Chest circumference range
    - waist_min/max: Waist circumference range
    - hip_min/max: Hip circumference range
    - height_min/max: Height range
    - fit_type: Fit type (tight, regular, loose)
    """
    # Get size chart and verify ownership via product
    size_chart = db.query(SizeChart).join(Product).filter(
        and_(
            SizeChart.id == chart_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not size_chart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Size chart {chart_id} not found"
        )

    # If updating size_name, check for duplicates
    if chart_data.size_name and chart_data.size_name != size_chart.size_name:
        existing = db.query(SizeChart).filter(
            and_(
                SizeChart.product_id == size_chart.product_id,
                SizeChart.size_name == chart_data.size_name,
                SizeChart.id != chart_id
            )
        ).first()

        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Size '{chart_data.size_name}' already exists for this product"
            )

    # Update fields (only if provided)
    update_data = chart_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(size_chart, field, value)

    db.commit()
    db.refresh(size_chart)

    return size_chart


@router.delete("/size-charts/{chart_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_size_chart(
    chart_id: UUID,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Delete a size chart

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - chart_id: UUID of the size chart
    """
    # Get size chart and verify ownership via product
    size_chart = db.query(SizeChart).join(Product).filter(
        and_(
            SizeChart.id == chart_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not size_chart:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Size chart {chart_id} not found"
        )

    db.delete(size_chart)
    db.commit()

    return None


# ===== Size Recommendation Endpoint =====

from app.schemas.product import SizeRecommendationRequest, SizeRecommendationResponse

@router.post("/products/{product_id}/recommend-size", response_model=SizeRecommendationResponse)
def recommend_size(
    product_id: UUID,
    request: SizeRecommendationRequest,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Get size recommendation for a product based on body measurements

    **Authentication**: Requires valid API key in X-API-Key header

    **Path Parameters**:
    - product_id: UUID of the product

    **Request Body**:
    - chest_circumference: Chest measurement in cm (optional)
    - waist_circumference: Waist measurement in cm (optional)
    - hip_circumference: Hip measurement in cm (optional)
    - height: Height in cm (optional)
    - fit_preference: Desired fit - tight, regular, loose (default: regular)

    **Returns**:
    - recommended_size: Best matching size
    - confidence: Confidence score (0-1)
    - size_scores: Fit scores for all sizes
    - fit_quality: Quality assessment (perfect, good, acceptable, poor)
    - alternative_sizes: Other sizes to consider
    """
    # Get product with size charts
    product = db.query(Product).filter(
        and_(
            Product.id == product_id,
            Product.brand_id == current_brand.id
        )
    ).first()

    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Product {product_id} not found"
        )

    # Get size charts for this product
    size_charts = db.query(SizeChart).filter(
        SizeChart.product_id == product_id
    ).order_by(SizeChart.display_order).all()

    if not size_charts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No size charts available for this product"
        )

    # Calculate fit scores for each size
    size_scores = {}
    for chart in size_charts:
        score = chart.matches_measurements(
            chest=request.chest_circumference,
            waist=request.waist_circumference,
            hip=request.hip_circumference,
            height=request.height
        )

        # Adjust score based on fit preference
        if request.fit_preference == "tight" and chart.fit_type == "tight":
            score *= 0.8  # Prefer tight fit
        elif request.fit_preference == "loose" and chart.fit_type == "loose":
            score *= 0.8  # Prefer loose fit
        elif request.fit_preference == "regular" and chart.fit_type == "regular":
            score *= 0.9  # Slight preference for regular

        size_scores[chart.size_name] = round(score, 2)

    # Find best size (lowest score = best fit)
    sorted_sizes = sorted(size_scores.items(), key=lambda x: x[1])
    best_size = sorted_sizes[0][0]
    best_score = sorted_sizes[0][1]

    # Calculate confidence (inverse of score, normalized)
    # Score of 0 = 100% confidence, score of 10+ = ~50% confidence
    confidence = max(0.5, 1.0 - (best_score / 20.0))
    confidence = min(1.0, confidence)

    # Determine fit quality
    if best_score < 1:
        fit_quality = "perfect"
    elif best_score < 3:
        fit_quality = "good"
    elif best_score < 6:
        fit_quality = "acceptable"
    else:
        fit_quality = "poor"

    # Get alternative sizes (next best options)
    alternatives = [s[0] for s in sorted_sizes[1:3] if s[1] < best_score + 5]

    # Get the fit type of the recommended size
    recommended_chart = next((c for c in size_charts if c.size_name == best_size), None)
    fit_type = recommended_chart.fit_type if recommended_chart else "regular"

    return SizeRecommendationResponse(
        recommended_size=best_size,
        confidence=round(confidence, 2),
        size_scores=size_scores,
        fit_quality=fit_quality,
        alternative_sizes=alternatives if alternatives else None,
        product_name=product.name,
        product_category=product.category,
        fit_type=fit_type
    )


@router.post("/recommend-size-bulk")
def recommend_size_bulk(
    measurements: dict,
    db: Session = Depends(get_db),
    current_brand: Brand = Depends(get_current_brand),
):
    """
    Get size recommendations for all products based on measurements

    **Authentication**: Requires valid API key in X-API-Key header

    **Request Body**:
    - chest_circumference: Chest measurement in cm (optional)
    - waist_circumference: Waist measurement in cm (optional)
    - hip_circumference: Hip measurement in cm (optional)
    - height: Height in cm (optional)
    - fit_preference: Desired fit (default: regular)
    - category: Filter by product category (optional)

    **Returns**: List of size recommendations for all matching products
    """
    # Get all products for this brand
    query = db.query(Product).filter(
        and_(
            Product.brand_id == current_brand.id,
            Product.is_active == True
        )
    )

    # Filter by category if provided
    if measurements.get("category"):
        query = query.filter(Product.category == measurements["category"])

    products = query.all()

    if not products:
        return {"recommendations": [], "total": 0}

    recommendations = []
    for product in products:
        size_charts = db.query(SizeChart).filter(
            SizeChart.product_id == product.id
        ).all()

        if not size_charts:
            continue

        # Calculate best size for this product
        size_scores = {}
        for chart in size_charts:
            score = chart.matches_measurements(
                chest=measurements.get("chest_circumference"),
                waist=measurements.get("waist_circumference"),
                hip=measurements.get("hip_circumference"),
                height=measurements.get("height")
            )
            size_scores[chart.size_name] = score

        sorted_sizes = sorted(size_scores.items(), key=lambda x: x[1])
        best_size = sorted_sizes[0][0]
        best_score = sorted_sizes[0][1]

        confidence = max(0.5, 1.0 - (best_score / 20.0))

        recommendations.append({
            "product_id": str(product.id),
            "product_name": product.name,
            "category": product.category,
            "recommended_size": best_size,
            "confidence": round(confidence, 2),
            "fit_quality": "perfect" if best_score < 1 else "good" if best_score < 3 else "acceptable" if best_score < 6 else "poor"
        })

    # Sort by confidence (highest first)
    recommendations.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "recommendations": recommendations,
        "total": len(recommendations)
    }
