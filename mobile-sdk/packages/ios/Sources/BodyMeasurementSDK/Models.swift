import Foundation

// MARK: - Fit Preference
public enum FitPreference: String, Codable {
    case tight
    case regular
    case loose
}

// MARK: - Measurement Response
public struct MeasurementResponse: Codable {
    public let shoulderWidth: Double
    public let chestWidth: Double
    public let waistWidth: Double
    public let hipWidth: Double
    public let inseam: Double
    public let armLength: Double
    public let confidenceScores: [String: Double]
    public let recommendedSize: String
    public let sizeProbabilities: [String: Double]
    public let processingTimeMs: Double

    enum CodingKeys: String, CodingKey {
        case shoulderWidth = "shoulder_width"
        case chestWidth = "chest_width"
        case waistWidth = "waist_width"
        case hipWidth = "hip_width"
        case inseam
        case armLength = "arm_length"
        case confidenceScores = "confidence_scores"
        case recommendedSize = "recommended_size"
        case sizeProbabilities = "size_probabilities"
        case processingTimeMs = "processing_time_ms"
    }
}

// MARK: - Product
public struct Product: Codable {
    public let id: String
    public let brandId: String
    public let name: String
    public let sku: String?
    public let category: String
    public let subcategory: String?
    public let gender: String?
    public let ageGroup: String?
    public let description: String?
    public let imageUrl: String?
    public let isActive: Bool
    public let sizeCharts: [SizeChart]
    public let createdAt: String
    public let updatedAt: String

    enum CodingKeys: String, CodingKey {
        case id, name, sku, category, subcategory, gender, description
        case brandId = "brand_id"
        case ageGroup = "age_group"
        case imageUrl = "image_url"
        case isActive = "is_active"
        case sizeCharts = "size_charts"
        case createdAt = "created_at"
        case updatedAt = "updated_at"
    }
}

// MARK: - Size Chart
public struct SizeChart: Codable {
    public let id: String
    public let productId: String
    public let sizeName: String
    public let chestMin: Double?
    public let chestMax: Double?
    public let waistMin: Double?
    public let waistMax: Double?
    public let hipMin: Double?
    public let hipMax: Double?
    public let fitType: String
    public let displayOrder: Int

    enum CodingKeys: String, CodingKey {
        case id, fitType
        case productId = "product_id"
        case sizeName = "size_name"
        case chestMin = "chest_min"
        case chestMax = "chest_max"
        case waistMin = "waist_min"
        case waistMax = "waist_max"
        case hipMin = "hip_min"
        case hipMax = "hip_max"
        case displayOrder = "display_order"
    }
}

// MARK: - Product List Response
public struct ProductListResponse: Codable {
    public let total: Int
    public let products: [Product]
}

// MARK: - SDK Error
public enum SDKError: Error {
    case invalidURL
    case networkError(Error)
    case noData
    case invalidResponse
    case apiError(Int, String)
    case decodingError(Error)

    public var localizedDescription: String {
        switch self {
        case .invalidURL:
            return "Invalid URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .noData:
            return "No data received"
        case .invalidResponse:
            return "Invalid response"
        case .apiError(let code, let message):
            return "API error (\(code)): \(message)"
        case .decodingError(let error):
            return "Decoding error: \(error.localizedDescription)"
        }
    }
}
