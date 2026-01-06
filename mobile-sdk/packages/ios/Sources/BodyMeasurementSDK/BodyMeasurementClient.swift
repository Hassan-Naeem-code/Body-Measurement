import Foundation
import UIKit

/// Main client for Body Measurement API
public class BodyMeasurementClient {
    private let apiKey: String
    private let baseURL: String
    private let session: URLSession
    private let debug: Bool

    public init(apiKey: String, baseURL: String = "http://localhost:8000", debug: Bool = false) {
        self.apiKey = apiKey
        self.baseURL = baseURL
        self.debug = debug
        self.session = URLSession.shared
    }

    /// Process single-person measurement
    public func processMeasurement(
        image: UIImage,
        productId: String? = nil,
        fitPreference: FitPreference = .regular,
        progress: ((Double) -> Void)? = nil,
        completion: @escaping (Result<MeasurementResponse, SDKError>) -> Void
    ) {
        guard let url = URL(string: "\(baseURL)/api/v1/measurements/process") else {
            completion(.failure(.invalidURL))
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")

        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        request.httpBody = createMultipartBody(
            image: image,
            boundary: boundary,
            productId: productId,
            fitPreference: fitPreference
        )

        log("Sending measurement request...")

        session.dataTask(with: request) { [weak self] data, response, error in
            if let error = error {
                completion(.failure(.networkError(error)))
                return
            }

            guard let data = data else {
                completion(.failure(.noData))
                return
            }

            guard let httpResponse = response as? HTTPURLResponse else {
                completion(.failure(.invalidResponse))
                return
            }

            guard httpResponse.statusCode == 200 else {
                let errorMsg = self?.parseErrorMessage(from: data) ?? "Unknown error"
                completion(.failure(.apiError(httpResponse.statusCode, errorMsg)))
                return
            }

            do {
                let result = try JSONDecoder().decode(MeasurementResponse.self, from: data)
                self?.log("Measurement successful", result)
                completion(.success(result))
            } catch {
                completion(.failure(.decodingError(error)))
            }
        }.resume()
    }

    /// Get list of products
    public func getProducts(
        skip: Int = 0,
        limit: Int = 10,
        category: String? = nil,
        completion: @escaping (Result<ProductListResponse, SDKError>) -> Void
    ) {
        var components = URLComponents(string: "\(baseURL)/api/v1/products")!
        var queryItems = [
            URLQueryItem(name: "skip", value: "\(skip)"),
            URLQueryItem(name: "limit", value: "\(limit)")
        ]
        if let category = category {
            queryItems.append(URLQueryItem(name: "category", value: category))
        }
        components.queryItems = queryItems

        guard let url = components.url else {
            completion(.failure(.invalidURL))
            return
        }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "X-API-Key")

        session.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(.networkError(error)))
                return
            }

            guard let data = data else {
                completion(.failure(.noData))
                return
            }

            do {
                let result = try JSONDecoder().decode(ProductListResponse.self, from: data)
                completion(.success(result))
            } catch {
                completion(.failure(.decodingError(error)))
            }
        }.resume()
    }

    // MARK: - Helper Methods

    private func createMultipartBody(
        image: UIImage,
        boundary: String,
        productId: String?,
        fitPreference: FitPreference
    ) -> Data {
        var body = Data()

        // Add image
        if let imageData = image.jpegData(compressionQuality: 0.8) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"file\"; filename=\"photo.jpg\"\r\n".data(using: .utf8)!)
            body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
            body.append(imageData)
            body.append("\r\n".data(using: .utf8)!)
        }

        // Add api_key
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"api_key\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(apiKey)\r\n".data(using: .utf8)!)

        // Add fit_preference
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"fit_preference\"\r\n\r\n".data(using: .utf8)!)
        body.append("\(fitPreference.rawValue)\r\n".data(using: .utf8)!)

        // Add product_id if provided
        if let productId = productId {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            body.append("Content-Disposition: form-data; name=\"product_id\"\r\n\r\n".data(using: .utf8)!)
            body.append("\(productId)\r\n".data(using: .utf8)!)
        }

        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        return body
    }

    private func parseErrorMessage(from data: Data) -> String {
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let detail = json["detail"] as? String {
            return detail
        }
        return String(data: data, encoding: .utf8) ?? "Unknown error"
    }

    private func log(_ message: String, _ data: Any? = nil) {
        if debug {
            print("[BodyMeasurementSDK] \(message)", data ?? "")
        }
    }
}
