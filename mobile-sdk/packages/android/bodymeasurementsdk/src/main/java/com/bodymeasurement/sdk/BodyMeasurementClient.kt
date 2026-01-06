package com.bodymeasurement.sdk

import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.asRequestBody
import com.google.gson.Gson
import java.io.File
import java.io.IOException

/**
 * Main client for Body Measurement API
 */
class BodyMeasurementClient(
    private val apiKey: String,
    private val baseURL: String = "http://localhost:8000",
    private val debug: Boolean = false
) {
    private val client = OkHttpClient()
    private val gson = Gson()

    /**
     * Process single-person body measurement
     *
     * Example:
     * ```
     * val client = BodyMeasurementClient(apiKey = "YOUR_API_KEY")
     * client.processMeasurement(
     *     imageFile = File("/path/to/image.jpg"),
     *     fitPreference = FitPreference.REGULAR
     * ) { result ->
     *     result.onSuccess { response ->
     *         println("Size: ${response.recommendedSize}")
     *     }.onFailure { error ->
     *         println("Error: ${error.message}")
     *     }
     * }
     * ```
     */
    fun processMeasurement(
        imageFile: File,
        productId: String? = null,
        fitPreference: FitPreference = FitPreference.REGULAR,
        callback: (Result<MeasurementResponse>) -> Unit
    ) {
        log("Processing measurement...")

        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "file",
                imageFile.name,
                imageFile.asRequestBody("image/jpeg".toMediaType())
            )
            .addFormDataPart("api_key", apiKey)
            .addFormDataPart("fit_preference", fitPreference.value)
            .apply {
                productId?.let { addFormDataPart("product_id", it) }
            }
            .build()

        val request = Request.Builder()
            .url("$baseURL/api/v1/measurements/process")
            .addHeader("X-API-Key", apiKey)
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!response.isSuccessful) {
                        val errorMsg = parseErrorMessage(response.body?.string())
                        callback(Result.failure(SDKException("API error (${response.code}): $errorMsg")))
                        return
                    }

                    try {
                        val body = response.body?.string()
                        val result = gson.fromJson(body, MeasurementResponse::class.java)
                        log("Measurement successful", result)
                        callback(Result.success(result))
                    } catch (e: Exception) {
                        callback(Result.failure(SDKException("Failed to parse response: ${e.message}")))
                    }
                }
            }

            override fun onFailure(call: Call, e: IOException) {
                callback(Result.failure(SDKException("Network error: ${e.message}")))
            }
        })
    }

    /**
     * Get list of products
     *
     * Example:
     * ```
     * client.getProducts(skip = 0, limit = 10) { result ->
     *     result.onSuccess { response ->
     *         println("Total products: ${response.total}")
     *         response.products.forEach { println(it.name) }
     *     }
     * }
     * ```
     */
    fun getProducts(
        skip: Int = 0,
        limit: Int = 10,
        category: String? = null,
        callback: (Result<ProductListResponse>) -> Unit
    ) {
        log("Fetching products...")

        val urlBuilder = HttpUrl.Builder()
            .scheme("http")
            .host(baseURL.removePrefix("http://").removePrefix("https://"))
            .addPathSegments("api/v1/products")
            .addQueryParameter("skip", skip.toString())
            .addQueryParameter("limit", limit.toString())

        category?.let { urlBuilder.addQueryParameter("category", it) }

        val request = Request.Builder()
            .url(urlBuilder.build())
            .addHeader("X-API-Key", apiKey)
            .get()
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onResponse(call: Call, response: Response) {
                response.use {
                    if (!response.isSuccessful) {
                        callback(Result.failure(SDKException("API error: ${response.code}")))
                        return
                    }

                    try {
                        val body = response.body?.string()
                        val result = gson.fromJson(body, ProductListResponse::class.java)
                        log("Products fetched", result)
                        callback(Result.success(result))
                    } catch (e: Exception) {
                        callback(Result.failure(SDKException("Failed to parse response: ${e.message}")))
                    }
                }
            }

            override fun onFailure(call: Call, e: IOException) {
                callback(Result.failure(SDKException("Network error: ${e.message}")))
            }
        })
    }

    private fun parseErrorMessage(body: String?): String {
        return try {
            val json = gson.fromJson(body, Map::class.java)
            json["detail"]?.toString() ?: "Unknown error"
        } catch (e: Exception) {
            body ?: "Unknown error"
        }
    }

    private fun log(message: String, data: Any? = null) {
        if (debug) {
            println("[BodyMeasurementSDK] $message ${data ?: ""}")
        }
    }
}
