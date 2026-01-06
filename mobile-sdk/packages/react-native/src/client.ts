/**
 * Body Measurement SDK Client
 */

import axios, { AxiosInstance, AxiosError } from 'axios';
import type {
  SDKConfig,
  MeasurementOptions,
  MeasurementResponse,
  MultiPersonMeasurementResponse,
  ProductListResponse,
  Product,
  ImageSource,
  SDKError,
  ProgressCallback,
} from './types';

/**
 * Main SDK Client for Body Measurement API
 */
export class BodyMeasurementClient {
  private client: AxiosInstance;
  private config: Required<SDKConfig>;

  /**
   * Initialize the SDK client
   *
   * @param config SDK configuration
   *
   * @example
   * ```typescript
   * const client = new BodyMeasurementClient({
   *   apiKey: 'your-api-key',
   *   baseURL: 'https://api.yourdomain.com',
   * });
   * ```
   */
  constructor(config: SDKConfig) {
    this.config = {
      apiKey: config.apiKey,
      baseURL: config.baseURL || 'http://localhost:8000',
      timeout: config.timeout || 30000,
      debug: config.debug || false,
    };

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'X-API-Key': this.config.apiKey,
      },
    });

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        return Promise.reject(this.handleError(error));
      }
    );
  }

  /**
   * Process a body image and get measurements (single person)
   *
   * @param image Image source (URI, file path, or base64)
   * @param options Measurement options
   * @param onProgress Optional progress callback
   * @returns Measurement response with size recommendation
   *
   * @example
   * ```typescript
   * const result = await client.processMeasurement(
   *   { uri: 'file:///path/to/image.jpg' },
   *   {
   *     productId: 'product-uuid',
   *     fitPreference: 'regular',
   *   },
   *   (progress) => console.log(`Upload: ${progress}%`)
   * );
   *
   * console.log('Recommended size:', result.recommended_size);
   * ```
   */
  async processMeasurement(
    image: ImageSource,
    options?: MeasurementOptions,
    onProgress?: ProgressCallback
  ): Promise<MeasurementResponse> {
    try {
      const formData = new FormData();

      // Add image file
      const imageFile = this.createImageFile(image);
      formData.append('file', imageFile as any);

      // Add API key (also in header, but required in form)
      formData.append('api_key', this.config.apiKey);

      // Add optional parameters
      if (options?.productId) {
        formData.append('product_id', options.productId);
      }
      if (options?.fitPreference) {
        formData.append('fit_preference', options.fitPreference);
      }

      this.log('Sending measurement request...', { options });

      const response = await this.client.post<MeasurementResponse>(
        '/api/v1/measurements/process',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            if (onProgress && progressEvent.total) {
              const progress = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              onProgress(progress);
            }
          },
        }
      );

      this.log('Measurement successful', response.data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Process a body image with multiple people
   *
   * @param image Image source
   * @param options Measurement options
   * @param onProgress Optional progress callback
   * @returns Multi-person measurement response
   *
   * @example
   * ```typescript
   * const result = await client.processMultiPersonMeasurement(
   *   { uri: 'file:///path/to/image.jpg' }
   * );
   *
   * console.log(`Found ${result.valid_people_count} valid people`);
   * result.measurements.forEach((person) => {
   *   console.log(`Person ${person.person_id}: ${person.recommended_size}`);
   * });
   * ```
   */
  async processMultiPersonMeasurement(
    image: ImageSource,
    options?: MeasurementOptions,
    onProgress?: ProgressCallback
  ): Promise<MultiPersonMeasurementResponse> {
    try {
      const formData = new FormData();

      const imageFile = this.createImageFile(image);
      formData.append('file', imageFile as any);
      formData.append('api_key', this.config.apiKey);

      if (options?.productId) {
        formData.append('product_id', options.productId);
      }
      if (options?.fitPreference) {
        formData.append('fit_preference', options.fitPreference);
      }

      this.log('Sending multi-person measurement request...', { options });

      const response = await this.client.post<MultiPersonMeasurementResponse>(
        '/api/v1/measurements/process-multi',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: (progressEvent) => {
            if (onProgress && progressEvent.total) {
              const progress = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              onProgress(progress);
            }
          },
        }
      );

      this.log('Multi-person measurement successful', response.data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get list of products
   *
   * @param skip Number of products to skip (pagination)
   * @param limit Number of products to return
   * @param category Optional category filter
   * @returns Paginated product list
   *
   * @example
   * ```typescript
   * const products = await client.getProducts(0, 10, 'tops');
   * console.log(`Found ${products.total} products`);
   * ```
   */
  async getProducts(
    skip: number = 0,
    limit: number = 10,
    category?: string
  ): Promise<ProductListResponse> {
    try {
      const params: any = { skip, limit };
      if (category) {
        params.category = category;
      }

      const response = await this.client.get<ProductListResponse>(
        '/api/v1/products',
        { params }
      );

      this.log('Products fetched', response.data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Get a specific product by ID
   *
   * @param productId Product UUID
   * @returns Product with size charts
   *
   * @example
   * ```typescript
   * const product = await client.getProduct('product-uuid');
   * console.log(product.name, product.size_charts);
   * ```
   */
  async getProduct(productId: string): Promise<Product> {
    try {
      const response = await this.client.get<Product>(
        `/api/v1/products/${productId}`
      );

      this.log('Product fetched', response.data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  /**
   * Create image file object from ImageSource
   */
  private createImageFile(image: ImageSource): any {
    const uri = image.uri;
    const type = image.type || 'image/jpeg';
    const name = image.name || `photo_${Date.now()}.jpg`;

    // React Native file object format
    return {
      uri,
      type,
      name,
    };
  }

  /**
   * Handle API errors and convert to SDKError
   */
  private handleError(error: any): SDKError {
    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError;
      const status = axiosError.response?.status;
      const message =
        (axiosError.response?.data as any)?.detail ||
        axiosError.message ||
        'Unknown error occurred';

      this.log('API Error', { status, message, data: axiosError.response?.data });

      // Create custom SDKError
      const sdkError = new (SDKError as any)(message, status, axiosError.response?.data);
      return sdkError;
    }

    this.log('Unknown Error', error);
    return new (SDKError as any)(
      error?.message || 'Unknown error occurred'
    );
  }

  /**
   * Debug logging
   */
  private log(message: string, data?: any): void {
    if (this.config.debug) {
      console.log(`[BodyMeasurementSDK] ${message}`, data || '');
    }
  }

  /**
   * Update API key
   */
  setApiKey(apiKey: string): void {
    this.config.apiKey = apiKey;
    this.client.defaults.headers['X-API-Key'] = apiKey;
  }

  /**
   * Get current configuration
   */
  getConfig(): Readonly<Required<SDKConfig>> {
    return { ...this.config };
  }
}
