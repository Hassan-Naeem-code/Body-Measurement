package com.bodymeasurement.sdk

import com.google.gson.annotations.SerializedName

// MARK: - Fit Preference
enum class FitPreference(val value: String) {
    TIGHT("tight"),
    REGULAR("regular"),
    LOOSE("loose")
}

// MARK: - Measurement Response
data class MeasurementResponse(
    @SerializedName("shoulder_width") val shoulderWidth: Double,
    @SerializedName("chest_width") val chestWidth: Double,
    @SerializedName("waist_width") val waistWidth: Double,
    @SerializedName("hip_width") val hipWidth: Double,
    @SerializedName("inseam") val inseam: Double,
    @SerializedName("arm_length") val armLength: Double,
    @SerializedName("confidence_scores") val confidenceScores: Map<String, Double>,
    @SerializedName("recommended_size") val recommendedSize: String,
    @SerializedName("size_probabilities") val sizeProbabilities: Map<String, Double>,
    @SerializedName("processing_time_ms") val processingTimeMs: Double
)

// MARK: - Product
data class Product(
    @SerializedName("id") val id: String,
    @SerializedName("brand_id") val brandId: String,
    @SerializedName("name") val name: String,
    @SerializedName("sku") val sku: String?,
    @SerializedName("category") val category: String,
    @SerializedName("subcategory") val subcategory: String?,
    @SerializedName("gender") val gender: String?,
    @SerializedName("age_group") val ageGroup: String?,
    @SerializedName("description") val description: String?,
    @SerializedName("image_url") val imageUrl: String?,
    @SerializedName("is_active") val isActive: Boolean,
    @SerializedName("size_charts") val sizeCharts: List<SizeChart>,
    @SerializedName("created_at") val createdAt: String,
    @SerializedName("updated_at") val updatedAt: String
)

// MARK: - Size Chart
data class SizeChart(
    @SerializedName("id") val id: String,
    @SerializedName("product_id") val productId: String,
    @SerializedName("size_name") val sizeName: String,
    @SerializedName("chest_min") val chestMin: Double?,
    @SerializedName("chest_max") val chestMax: Double?,
    @SerializedName("waist_min") val waistMin: Double?,
    @SerializedName("waist_max") val waistMax: Double?,
    @SerializedName("hip_min") val hipMin: Double?,
    @SerializedName("hip_max") val hipMax: Double?,
    @SerializedName("fit_type") val fitType: String,
    @SerializedName("display_order") val displayOrder: Int
)

// MARK: - Product List Response
data class ProductListResponse(
    @SerializedName("total") val total: Int,
    @SerializedName("products") val products: List<Product>
)

// MARK: - SDK Exception
class SDKException(message: String) : Exception(message)
