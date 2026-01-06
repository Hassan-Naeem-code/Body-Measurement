/**
 * Type definitions for Body Measurement SDK
 */

/**
 * SDK Configuration
 */
export interface SDKConfig {
  /** Your API key for authentication */
  apiKey: string;
  /** API base URL (default: production URL) */
  baseURL?: string;
  /** Request timeout in milliseconds (default: 30000) */
  timeout?: number;
  /** Enable debug logging (default: false) */
  debug?: boolean;
}

/**
 * Fit preference for size recommendations
 */
export type FitPreference = 'tight' | 'regular' | 'loose';

/**
 * Measurement request options
 */
export interface MeasurementOptions {
  /** Optional product ID for product-specific sizing */
  productId?: string;
  /** Fit preference (default: 'regular') */
  fitPreference?: FitPreference;
}

/**
 * Body measurements response
 */
export interface MeasurementResponse {
  /** Shoulder width in cm */
  shoulder_width: number;
  /** Chest width in cm */
  chest_width: number;
  /** Waist width in cm */
  waist_width: number;
  /** Hip width in cm */
  hip_width: number;
  /** Inseam length in cm */
  inseam: number;
  /** Arm length in cm */
  arm_length: number;
  /** Confidence scores for each measurement */
  confidence_scores: Record<string, number>;
  /** Recommended size */
  recommended_size: string;
  /** Probability distribution across all sizes */
  size_probabilities: Record<string, number>;
  /** Processing time in milliseconds */
  processing_time_ms: number;
}

/**
 * Multi-person measurement response (single person)
 */
export interface PersonMeasurement {
  /** Person ID in the image */
  person_id: number;
  /** Detection confidence */
  detection_confidence: number;
  /** Whether person passed validation */
  is_valid: boolean;
  /** Missing body parts (if invalid) */
  missing_parts: string[];
  /** Validation confidence */
  validation_confidence: number;
  /** Confidence scores per body part */
  body_part_confidences: Record<string, number>;
  /** Measurements (null if invalid) */
  shoulder_width: number | null;
  chest_width: number | null;
  waist_width: number | null;
  hip_width: number | null;
  inseam: number | null;
  arm_length: number | null;
  /** Recommended size (null if invalid) */
  recommended_size: string | null;
  /** Size probabilities (null if invalid) */
  size_probabilities: Record<string, number> | null;
}

/**
 * Multi-person measurement response
 */
export interface MultiPersonMeasurementResponse {
  /** Total people detected */
  total_people_detected: number;
  /** Valid people count */
  valid_people_count: number;
  /** Invalid people count */
  invalid_people_count: number;
  /** Measurements for all valid people */
  measurements: PersonMeasurement[];
  /** Processing time in milliseconds */
  processing_time_ms: number;
  /** Processing metadata */
  processing_metadata: Record<string, string>;
}

/**
 * Image source for measurement
 */
export interface ImageSource {
  /** Image URI (file:// or content://) */
  uri: string;
  /** MIME type (image/jpeg, image/png, image/webp) */
  type?: string;
  /** File name */
  name?: string;
}

/**
 * Product size chart
 */
export interface SizeChart {
  /** Size chart ID */
  id: string;
  /** Product ID */
  product_id: string;
  /** Size name (XS, S, M, L, etc.) */
  size_name: string;
  /** Min/max measurements */
  chest_min?: number;
  chest_max?: number;
  waist_min?: number;
  waist_max?: number;
  hip_min?: number;
  hip_max?: number;
  height_min?: number;
  height_max?: number;
  inseam_min?: number;
  inseam_max?: number;
  shoulder_width_min?: number;
  shoulder_width_max?: number;
  arm_length_min?: number;
  arm_length_max?: number;
  weight_min?: number;
  weight_max?: number;
  /** Fit type */
  fit_type: string;
  /** Display order */
  display_order: number;
  /** Created timestamp */
  created_at: string;
}

/**
 * Product with size charts
 */
export interface Product {
  /** Product ID */
  id: string;
  /** Brand ID */
  brand_id: string;
  /** Product name */
  name: string;
  /** SKU */
  sku?: string;
  /** Category */
  category: string;
  /** Subcategory */
  subcategory?: string;
  /** Gender target */
  gender?: string;
  /** Age group target */
  age_group?: string;
  /** Description */
  description?: string;
  /** Image URL */
  image_url?: string;
  /** Active status */
  is_active: boolean;
  /** Size charts */
  size_charts: SizeChart[];
  /** Created timestamp */
  created_at: string;
  /** Updated timestamp */
  updated_at: string;
}

/**
 * Paginated product list
 */
export interface ProductListResponse {
  /** Total products */
  total: number;
  /** Products */
  products: Product[];
}

/**
 * SDK Error
 */
export class SDKError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public response?: any
  ) {
    super(message);
    this.name = 'SDKError';
  }
}

/**
 * Camera options for image capture
 */
export interface CameraOptions {
  /** Camera type (front or back) */
  cameraType?: 'front' | 'back';
  /** Image quality (0-100) */
  quality?: number;
  /** Max image width */
  maxWidth?: number;
  /** Max image height */
  maxHeight?: number;
}

/**
 * Progress callback for uploads
 */
export type ProgressCallback = (progress: number) => void;
