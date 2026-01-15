# FitWhisperer Widget

A lightweight, embeddable JavaScript widget for adding AI-powered size recommendations to any e-commerce website.

## Features

- Drag-and-drop image upload
- Live camera capture
- AI-powered body measurement extraction
- Personalized size recommendations
- Fit preference selection (tight, regular, loose)
- Dark/light theme support
- Fully customizable styling
- Mobile-responsive design
- No dependencies (vanilla JavaScript)

## Quick Start

### Option 1: Script Tag (Easiest)

Add this script tag to your HTML, replacing `YOUR_API_KEY` with your actual API key:

```html
<script
  src="https://cdn.fitwhisperer.io/widget.js"
  data-bm-api-key="YOUR_API_KEY"
  data-bm-product-id="OPTIONAL_PRODUCT_UUID"
></script>
```

The widget button will appear automatically!

### Option 2: Manual Initialization

```html
<script src="https://cdn.fitwhisperer.io/widget.js"></script>
<script>
  const widget = new BodyMeasurementWidget({
    apiKey: 'YOUR_API_KEY',
    productId: 'PRODUCT_UUID', // Optional
    onSizeRecommendation: function(recommendation) {
      console.log('Recommended size:', recommendation.recommendedSize);
      // Update your product page
    }
  });
</script>
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apiKey` | string | null | **Required.** Your API key |
| `apiUrl` | string | `https://api.fitwhisperer.io` | API endpoint URL |
| `productId` | string | null | Product UUID for size recommendations |
| `theme` | string | `'light'` | Theme: `'light'` or `'dark'` |
| `primaryColor` | string | `'#6366f1'` | Primary button/accent color |
| `position` | string | `'bottom-right'` | Button position: `'bottom-right'`, `'bottom-left'`, `'top-right'`, `'top-left'` |
| `buttonText` | string | `'Find My Size'` | Text on the widget button |
| `locale` | string | `'en'` | Language locale |
| `showConfidence` | boolean | `true` | Show confidence percentage |
| `onMeasurement` | function | null | Callback when measurements are ready |
| `onSizeRecommendation` | function | null | Callback when size recommendation is ready |
| `onError` | function | null | Callback when an error occurs |

## Data Attributes

When using the script tag approach, you can configure the widget using data attributes:

```html
<script
  src="https://cdn.fitwhisperer.io/widget.js"
  data-bm-api-key="YOUR_API_KEY"
  data-bm-product-id="PRODUCT_UUID"
  data-bm-theme="light"
  data-bm-color="#6366f1"
  data-bm-position="bottom-right"
  data-bm-button-text="Find My Size"
></script>
```

## JavaScript API

### Methods

```javascript
// Create widget instance
const widget = new BodyMeasurementWidget({ apiKey: 'YOUR_API_KEY' });

// Open the widget modal
widget.open();

// Close the widget modal
widget.close();

// Update product ID (useful for single-page apps)
widget.setProductId('new-product-uuid');

// Destroy the widget (remove from DOM)
widget.destroy();
```

### Callbacks

```javascript
const widget = new BodyMeasurementWidget({
  apiKey: 'YOUR_API_KEY',
  productId: 'PRODUCT_UUID',

  // Called when measurements are extracted from the image
  onMeasurement: function(result) {
    console.log('Total people detected:', result.total_people_detected);
    console.log('Valid measurements:', result.valid_people_count);

    const person = result.measurements.find(p => p.is_valid);
    if (person) {
      console.log('Chest:', person.chest_circumference, 'cm');
      console.log('Waist:', person.waist_circumference, 'cm');
      console.log('Hip:', person.hip_circumference, 'cm');
    }
  },

  // Called when size recommendation is ready
  onSizeRecommendation: function(recommendation) {
    console.log('Recommended size:', recommendation.recommended_size);
    console.log('Confidence:', recommendation.confidence);
    console.log('Fit quality:', recommendation.fit_quality);
    console.log('Alternative sizes:', recommendation.alternative_sizes);

    // Update your product page
    document.querySelector('.size-selector').value = recommendation.recommended_size;
  },

  // Called when an error occurs
  onError: function(error) {
    console.error('Widget error:', error.message);
  }
});
```

## Response Objects

### Measurement Result

```javascript
{
  total_people_detected: 1,
  valid_people_count: 1,
  invalid_people_count: 0,
  measurements: [{
    person_id: 0,
    is_valid: true,
    detection_confidence: 0.95,
    validation_confidence: 0.92,

    // Demographics
    gender: 'female',
    age_group: 'adult',
    gender_confidence: 0.89,
    age_confidence: 0.87,

    // Circumferences (95%+ accuracy)
    chest_circumference: 88.5,
    waist_circumference: 72.3,
    hip_circumference: 96.8,

    // Additional
    estimated_height_cm: 165.2,
    recommended_size: 'M',
    size_probabilities: {
      'XS': 0.05,
      'S': 0.25,
      'M': 0.55,
      'L': 0.15
    }
  }],
  processing_time_ms: 1234.5
}
```

### Size Recommendation

```javascript
{
  recommended_size: 'M',
  confidence: 0.92,
  fit_quality: 'good', // 'perfect', 'good', 'acceptable', 'poor'
  alternative_sizes: ['S', 'L'],
  size_scores: {
    'XS': 8.5,
    'S': 2.3,
    'M': 0.5,
    'L': 3.1,
    'XL': 7.8
  },
  product_name: 'Classic T-Shirt',
  product_category: 'tops',
  fit_type: 'regular'
}
```

## Styling Customization

### CSS Variables

The widget uses CSS variables that you can override:

```css
:root {
  --bm-primary: #6366f1; /* Primary color */
}
```

### Custom Styles

Add your own styles to customize the appearance:

```css
/* Custom button styling */
.bm-widget-button {
  border-radius: 8px !important;
  font-size: 16px !important;
}

/* Custom modal styling */
.bm-modal {
  max-width: 600px !important;
}

/* Dark theme adjustments */
.bm-modal.dark {
  background: #0f172a !important;
}
```

## Integration Examples

### Shopify

```liquid
<!-- In your product.liquid template -->
<script
  src="https://cdn.fitwhisperer.io/widget.js"
  data-bm-api-key="{{ settings.body_measurement_api_key }}"
  data-bm-product-id="{{ product.metafields.fitwhisperer.product_id }}"
></script>
```

### WooCommerce

```php
// In your theme's functions.php
function add_body_measurement_widget() {
  if (is_product()) {
    global $product;
    $product_id = get_post_meta($product->get_id(), '_bm_product_id', true);
    ?>
    <script
      src="https://cdn.fitwhisperer.io/widget.js"
      data-bm-api-key="<?php echo BODY_MEASUREMENT_API_KEY; ?>"
      data-bm-product-id="<?php echo esc_attr($product_id); ?>"
    ></script>
    <?php
  }
}
add_action('wp_footer', 'add_body_measurement_widget');
```

### React

```jsx
import { useEffect } from 'react';

function ProductPage({ productId }) {
  useEffect(() => {
    const widget = new window.BodyMeasurementWidget({
      apiKey: process.env.REACT_APP_BM_API_KEY,
      productId: productId,
      onSizeRecommendation: (rec) => {
        // Handle recommendation
      }
    });

    return () => widget.destroy();
  }, [productId]);

  return <div>...</div>;
}
```

### Vue.js

```vue
<script setup>
import { onMounted, onUnmounted, ref } from 'vue';

const props = defineProps(['productId']);
let widget = ref(null);

onMounted(() => {
  widget.value = new window.BodyMeasurementWidget({
    apiKey: import.meta.env.VITE_BM_API_KEY,
    productId: props.productId,
  });
});

onUnmounted(() => {
  widget.value?.destroy();
});
</script>
```

## Browser Support

- Chrome 60+
- Firefox 55+
- Safari 11+
- Edge 79+
- iOS Safari 11+
- Android Chrome 60+

## File Size

- Minified: ~15KB
- Gzipped: ~5KB

## Support

- Documentation: https://docs.fitwhisperer.io/widget
- Email: support@fitwhisperer.io
- GitHub Issues: https://github.com/your-org/fitwhisperer-widget/issues

## License

MIT License
