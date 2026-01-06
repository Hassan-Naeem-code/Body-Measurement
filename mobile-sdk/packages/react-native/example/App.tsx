/**
 * Example App demonstrating Body Measurement SDK usage
 */

import React, { useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  Text,
  View,
  Button,
  ScrollView,
  Image,
  ActivityIndicator,
  Alert,
} from 'react-native';
import {
  BodyMeasurementClient,
  useMeasurement,
  useCamera,
  useProducts,
  type FitPreference,
} from '@body-measurement/react-native-sdk';

// Initialize SDK client
const client = new BodyMeasurementClient({
  apiKey: 'YOUR_API_KEY_HERE', // Replace with your API key
  baseURL: 'http://localhost:8000', // Replace with your API URL
  debug: true,
});

function App(): JSX.Element {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [fitPreference, setFitPreference] = useState<FitPreference>('regular');

  // Hooks
  const { loading, error, data, progress, processMeasurement } = useMeasurement(client);
  const { captureImage, selectFromGallery } = useCamera();
  const { products, loading: loadingProducts } = useProducts(client, 0, 5);

  // Handle camera capture
  const handleCapture = async () => {
    try {
      const image = await captureImage({
        cameraType: 'back',
        quality: 80,
        maxWidth: 1920,
        maxHeight: 1920,
      });

      if (image) {
        setSelectedImage(image.uri);
        Alert.alert('Success', 'Photo captured! Tap "Process Measurement" to analyze.');
      }
    } catch (err) {
      Alert.alert('Error', (err as Error).message);
    }
  };

  // Handle gallery selection
  const handleGallery = async () => {
    try {
      const image = await selectFromGallery({
        quality: 80,
        maxWidth: 1920,
        maxHeight: 1920,
      });

      if (image) {
        setSelectedImage(image.uri);
        Alert.alert('Success', 'Photo selected! Tap "Process Measurement" to analyze.');
      }
    } catch (err) {
      Alert.alert('Error', (err as Error).message);
    }
  };

  // Handle measurement processing
  const handleProcess = async () => {
    if (!selectedImage) {
      Alert.alert('Error', 'Please capture or select a photo first');
      return;
    }

    try {
      await processMeasurement(
        { uri: selectedImage },
        { fitPreference }
      );
    } catch (err) {
      Alert.alert('Error', (err as Error).message);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Body Measurement SDK</Text>
          <Text style={styles.subtitle}>AI-Powered Size Recommendations</Text>
        </View>

        {/* Image Preview */}
        {selectedImage && (
          <View style={styles.imageContainer}>
            <Image source={{ uri: selectedImage }} style={styles.image} />
          </View>
        )}

        {/* Camera Buttons */}
        <View style={styles.buttonGroup}>
          <Button title="Take Photo" onPress={handleCapture} />
          <View style={styles.buttonSpacer} />
          <Button title="Choose from Gallery" onPress={handleGallery} />
        </View>

        {/* Fit Preference Selector */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Fit Preference:</Text>
          <View style={styles.buttonGroup}>
            <Button
              title="Tight"
              onPress={() => setFitPreference('tight')}
              color={fitPreference === 'tight' ? '#007AFF' : '#8E8E93'}
            />
            <Button
              title="Regular"
              onPress={() => setFitPreference('regular')}
              color={fitPreference === 'regular' ? '#007AFF' : '#8E8E93'}
            />
            <Button
              title="Loose"
              onPress={() => setFitPreference('loose')}
              color={fitPreference === 'loose' ? '#007AFF' : '#8E8E93'}
            />
          </View>
        </View>

        {/* Process Button */}
        <View style={styles.processButton}>
          <Button
            title="Process Measurement"
            onPress={handleProcess}
            disabled={!selectedImage || loading}
          />
        </View>

        {/* Loading Indicator */}
        {loading && (
          <View style={styles.section}>
            <ActivityIndicator size="large" color="#007AFF" />
            <Text style={styles.progressText}>
              Processing... {progress}%
            </Text>
          </View>
        )}

        {/* Error Display */}
        {error && (
          <View style={styles.errorContainer}>
            <Text style={styles.errorText}>Error: {error.message}</Text>
          </View>
        )}

        {/* Results Display */}
        {data && (
          <View style={styles.resultsContainer}>
            <Text style={styles.resultsTitle}>Results:</Text>

            <View style={styles.resultRow}>
              <Text style={styles.resultLabel}>Recommended Size:</Text>
              <Text style={styles.resultValue}>{data.recommended_size}</Text>
            </View>

            <View style={styles.resultRow}>
              <Text style={styles.resultLabel}>Confidence:</Text>
              <Text style={styles.resultValue}>
                {Math.round(
                  data.size_probabilities[data.recommended_size] * 100
                )}%
              </Text>
            </View>

            <Text style={styles.measurementsTitle}>Measurements (cm):</Text>
            <View style={styles.measurementGrid}>
              <View style={styles.measurementItem}>
                <Text style={styles.measurementLabel}>Chest:</Text>
                <Text style={styles.measurementValue}>{data.chest_width.toFixed(1)}</Text>
              </View>
              <View style={styles.measurementItem}>
                <Text style={styles.measurementLabel}>Waist:</Text>
                <Text style={styles.measurementValue}>{data.waist_width.toFixed(1)}</Text>
              </View>
              <View style={styles.measurementItem}>
                <Text style={styles.measurementLabel}>Hip:</Text>
                <Text style={styles.measurementValue}>{data.hip_width.toFixed(1)}</Text>
              </View>
              <View style={styles.measurementItem}>
                <Text style={styles.measurementLabel}>Inseam:</Text>
                <Text style={styles.measurementValue}>{data.inseam.toFixed(1)}</Text>
              </View>
            </View>

            <Text style={styles.processingTime}>
              Processing time: {data.processing_time_ms.toFixed(0)}ms
            </Text>
          </View>
        )}

        {/* Products List */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Available Products:</Text>
          {loadingProducts ? (
            <ActivityIndicator size="small" />
          ) : (
            products.map((product) => (
              <View key={product.id} style={styles.productItem}>
                <Text style={styles.productName}>{product.name}</Text>
                <Text style={styles.productCategory}>{product.category}</Text>
              </View>
            ))
          )}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#000',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  imageContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  image: {
    width: 300,
    height: 400,
    borderRadius: 10,
    backgroundColor: '#E0E0E0',
  },
  buttonGroup: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  buttonSpacer: {
    width: 10,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 10,
    color: '#000',
  },
  processButton: {
    marginBottom: 20,
  },
  progressText: {
    textAlign: 'center',
    marginTop: 10,
    color: '#666',
  },
  errorContainer: {
    backgroundColor: '#FFE6E6',
    padding: 15,
    borderRadius: 8,
    marginBottom: 20,
  },
  errorText: {
    color: '#D32F2F',
  },
  resultsContainer: {
    backgroundColor: '#FFF',
    padding: 20,
    borderRadius: 10,
    marginBottom: 20,
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 15,
    color: '#000',
  },
  resultRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  resultLabel: {
    fontSize: 16,
    color: '#666',
  },
  resultValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
  },
  measurementsTitle: {
    fontSize: 16,
    fontWeight: '600',
    marginTop: 15,
    marginBottom: 10,
    color: '#000',
  },
  measurementGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  measurementItem: {
    width: '48%',
    backgroundColor: '#F5F5F5',
    padding: 10,
    borderRadius: 8,
    marginBottom: 10,
  },
  measurementLabel: {
    fontSize: 14,
    color: '#666',
  },
  measurementValue: {
    fontSize: 18,
    fontWeight: '600',
    color: '#007AFF',
    marginTop: 4,
  },
  processingTime: {
    fontSize: 12,
    color: '#999',
    marginTop: 15,
    textAlign: 'center',
  },
  productItem: {
    backgroundColor: '#FFF',
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
  },
  productName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#000',
  },
  productCategory: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
});

export default App;
