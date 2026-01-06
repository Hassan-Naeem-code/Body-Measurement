/**
 * React Hooks for Body Measurement SDK
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  BodyMeasurementClient,
  ImageSource,
  MeasurementOptions,
  MeasurementResponse,
  MultiPersonMeasurementResponse,
  Product,
  CameraOptions,
} from './types';

/**
 * Hook state for measurement operations
 */
interface UseMeasurementState {
  /** Loading state */
  loading: boolean;
  /** Error state */
  error: Error | null;
  /** Measurement result */
  data: MeasurementResponse | null;
  /** Upload progress (0-100) */
  progress: number;
}

/**
 * Hook for processing measurements
 *
 * @param client SDK client instance
 * @returns Measurement state and process function
 *
 * @example
 * ```typescript
 * const { loading, error, data, progress, processMeasurement } = useMeasurement(client);
 *
 * const handleCapture = async (imageUri: string) => {
 *   const result = await processMeasurement(
 *     { uri: imageUri },
 *     { fitPreference: 'regular' }
 *   );
 *   console.log('Size:', result.recommended_size);
 * };
 * ```
 */
export function useMeasurement(client: BodyMeasurementClient) {
  const [state, setState] = useState<UseMeasurementState>({
    loading: false,
    error: null,
    data: null,
    progress: 0,
  });

  const processMeasurement = useCallback(
    async (image: ImageSource, options?: MeasurementOptions) => {
      setState({ loading: true, error: null, data: null, progress: 0 });

      try {
        const result = await client.processMeasurement(
          image,
          options,
          (progress) => {
            setState((prev) => ({ ...prev, progress }));
          }
        );

        setState({ loading: false, error: null, data: result, progress: 100 });
        return result;
      } catch (error) {
        const err = error as Error;
        setState({ loading: false, error: err, data: null, progress: 0 });
        throw err;
      }
    },
    [client]
  );

  const reset = useCallback(() => {
    setState({ loading: false, error: null, data: null, progress: 0 });
  }, []);

  return {
    ...state,
    processMeasurement,
    reset,
  };
}

/**
 * Hook state for multi-person measurement
 */
interface UseMultiPersonMeasurementState {
  loading: boolean;
  error: Error | null;
  data: MultiPersonMeasurementResponse | null;
  progress: number;
}

/**
 * Hook for processing multi-person measurements
 *
 * @param client SDK client instance
 * @returns Multi-person measurement state and process function
 */
export function useMultiPersonMeasurement(client: BodyMeasurementClient) {
  const [state, setState] = useState<UseMultiPersonMeasurementState>({
    loading: false,
    error: null,
    data: null,
    progress: 0,
  });

  const processMultiPersonMeasurement = useCallback(
    async (image: ImageSource, options?: MeasurementOptions) => {
      setState({ loading: false, error: null, data: null, progress: 0 });

      try {
        const result = await client.processMultiPersonMeasurement(
          image,
          options,
          (progress) => {
            setState((prev) => ({ ...prev, progress }));
          }
        );

        setState({ loading: false, error: null, data: result, progress: 100 });
        return result;
      } catch (error) {
        const err = error as Error;
        setState({ loading: false, error: err, data: null, progress: 0 });
        throw err;
      }
    },
    [client]
  );

  const reset = useCallback(() => {
    setState({ loading: false, error: null, data: null, progress: 0 });
  }, []);

  return {
    ...state,
    processMultiPersonMeasurement,
    reset,
  };
}

/**
 * Hook state for products
 */
interface UseProductsState {
  loading: boolean;
  error: Error | null;
  products: Product[];
  total: number;
}

/**
 * Hook for fetching products
 *
 * @param client SDK client instance
 * @param skip Number of products to skip
 * @param limit Number of products to fetch
 * @param category Optional category filter
 * @returns Products state and refresh function
 *
 * @example
 * ```typescript
 * const { loading, error, products, total, refresh } = useProducts(client, 0, 10, 'tops');
 *
 * return (
 *   <FlatList
 *     data={products}
 *     renderItem={({ item }) => <ProductCard product={item} />}
 *   />
 * );
 * ```
 */
export function useProducts(
  client: BodyMeasurementClient,
  skip: number = 0,
  limit: number = 10,
  category?: string
) {
  const [state, setState] = useState<UseProductsState>({
    loading: true,
    error: null,
    products: [],
    total: 0,
  });

  const fetchProducts = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const result = await client.getProducts(skip, limit, category);
      setState({
        loading: false,
        error: null,
        products: result.products,
        total: result.total,
      });
    } catch (error) {
      const err = error as Error;
      setState({ loading: false, error: err, products: [], total: 0 });
    }
  }, [client, skip, limit, category]);

  useEffect(() => {
    fetchProducts();
  }, [fetchProducts]);

  return {
    ...state,
    refresh: fetchProducts,
  };
}

/**
 * Hook state for single product
 */
interface UseProductState {
  loading: boolean;
  error: Error | null;
  product: Product | null;
}

/**
 * Hook for fetching a single product
 *
 * @param client SDK client instance
 * @param productId Product ID
 * @returns Product state and refresh function
 */
export function useProduct(client: BodyMeasurementClient, productId: string) {
  const [state, setState] = useState<UseProductState>({
    loading: true,
    error: null,
    product: null,
  });

  const fetchProduct = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const result = await client.getProduct(productId);
      setState({ loading: false, error: null, product: result });
    } catch (error) {
      const err = error as Error;
      setState({ loading: false, error: err, product: null });
    }
  }, [client, productId]);

  useEffect(() => {
    fetchProduct();
  }, [fetchProduct]);

  return {
    ...state,
    refresh: fetchProduct,
  };
}

/**
 * Hook for camera functionality
 * Note: Requires react-native-image-picker to be installed
 *
 * @example
 * ```typescript
 * const { captureImage, selectFromGallery } = useCamera();
 *
 * const handleCapture = async () => {
 *   const image = await captureImage({ cameraType: 'back', quality: 80 });
 *   if (image) {
 *     // Process measurement
 *     const result = await processMeasurement(image);
 *   }
 * };
 * ```
 */
export function useCamera() {
  const captureImage = useCallback(
    async (options?: CameraOptions): Promise<ImageSource | null> => {
      try {
        // Try to import react-native-image-picker
        const ImagePicker = require('react-native-image-picker');

        const result = await ImagePicker.launchCamera({
          mediaType: 'photo',
          cameraType: options?.cameraType || 'back',
          quality: (options?.quality || 80) / 100,
          maxWidth: options?.maxWidth,
          maxHeight: options?.maxHeight,
        });

        if (result.didCancel || !result.assets?.[0]) {
          return null;
        }

        const asset = result.assets[0];
        return {
          uri: asset.uri!,
          type: asset.type || 'image/jpeg',
          name: asset.fileName || `photo_${Date.now()}.jpg`,
        };
      } catch (error) {
        console.error('Camera error:', error);
        throw new Error(
          'Failed to capture image. Make sure react-native-image-picker is installed.'
        );
      }
    },
    []
  );

  const selectFromGallery = useCallback(
    async (options?: CameraOptions): Promise<ImageSource | null> => {
      try {
        const ImagePicker = require('react-native-image-picker');

        const result = await ImagePicker.launchImageLibrary({
          mediaType: 'photo',
          quality: (options?.quality || 80) / 100,
          maxWidth: options?.maxWidth,
          maxHeight: options?.maxHeight,
        });

        if (result.didCancel || !result.assets?.[0]) {
          return null;
        }

        const asset = result.assets[0];
        return {
          uri: asset.uri!,
          type: asset.type || 'image/jpeg',
          name: asset.fileName || `photo_${Date.now()}.jpg`,
        };
      } catch (error) {
        console.error('Gallery error:', error);
        throw new Error(
          'Failed to select image. Make sure react-native-image-picker is installed.'
        );
      }
    },
    []
  );

  return {
    captureImage,
    selectFromGallery,
  };
}
