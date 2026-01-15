/**
 * FitWhisperer Widget
 * Embeddable widget for e-commerce size recommendations
 * Version: 1.0.0
 */

(function (window, document) {
  'use strict';

  // Default configuration
  const DEFAULT_CONFIG = {
    apiKey: null,
    apiUrl: 'https://api.fitwhisperer.io',
    productId: null,
    theme: 'light',
    primaryColor: '#6366f1',
    position: 'bottom-right',
    buttonText: 'Find My Size',
    locale: 'en',
    showConfidence: true,
    onMeasurement: null,
    onSizeRecommendation: null,
    onError: null,
  };

  // Styles
  const STYLES = `
    .bm-widget-button {
      position: fixed;
      z-index: 999998;
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 12px 20px;
      border: none;
      border-radius: 50px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      transition: all 0.2s ease;
    }
    .bm-widget-button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    }
    .bm-widget-button.bottom-right {
      bottom: 20px;
      right: 20px;
    }
    .bm-widget-button.bottom-left {
      bottom: 20px;
      left: 20px;
    }
    .bm-widget-button.top-right {
      top: 20px;
      right: 20px;
    }
    .bm-widget-button.top-left {
      top: 20px;
      left: 20px;
    }
    .bm-widget-button svg {
      width: 20px;
      height: 20px;
    }

    .bm-modal-overlay {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.5);
      z-index: 999999;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
      opacity: 0;
      visibility: hidden;
      transition: all 0.3s ease;
    }
    .bm-modal-overlay.open {
      opacity: 1;
      visibility: visible;
    }

    .bm-modal {
      background: white;
      border-radius: 16px;
      width: 100%;
      max-width: 480px;
      max-height: 90vh;
      overflow: hidden;
      transform: scale(0.9) translateY(20px);
      transition: transform 0.3s ease;
    }
    .bm-modal-overlay.open .bm-modal {
      transform: scale(1) translateY(0);
    }
    .bm-modal.dark {
      background: #1f2937;
      color: white;
    }

    .bm-modal-header {
      padding: 20px;
      border-bottom: 1px solid #e5e7eb;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .bm-modal.dark .bm-modal-header {
      border-color: #374151;
    }
    .bm-modal-header h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    }
    .bm-close-btn {
      background: none;
      border: none;
      padding: 8px;
      cursor: pointer;
      border-radius: 8px;
      color: #6b7280;
      transition: all 0.2s;
    }
    .bm-close-btn:hover {
      background: #f3f4f6;
      color: #111827;
    }
    .bm-modal.dark .bm-close-btn:hover {
      background: #374151;
      color: white;
    }

    .bm-modal-content {
      padding: 20px;
      max-height: calc(90vh - 150px);
      overflow-y: auto;
    }

    .bm-step {
      display: none;
    }
    .bm-step.active {
      display: block;
    }

    .bm-upload-area {
      border: 2px dashed #d1d5db;
      border-radius: 12px;
      padding: 40px 20px;
      text-align: center;
      cursor: pointer;
      transition: all 0.2s;
    }
    .bm-upload-area:hover {
      border-color: var(--bm-primary);
      background: rgba(99, 102, 241, 0.05);
    }
    .bm-upload-area.dragover {
      border-color: var(--bm-primary);
      background: rgba(99, 102, 241, 0.1);
    }
    .bm-upload-icon {
      width: 48px;
      height: 48px;
      margin: 0 auto 16px;
      color: #9ca3af;
    }
    .bm-upload-text {
      color: #6b7280;
      margin-bottom: 16px;
    }
    .bm-upload-text strong {
      color: #111827;
    }
    .bm-modal.dark .bm-upload-text strong {
      color: white;
    }

    .bm-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 24px;
      border: none;
      border-radius: 10px;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      font-family: inherit;
    }
    .bm-btn-primary {
      background: var(--bm-primary);
      color: white;
    }
    .bm-btn-primary:hover:not(:disabled) {
      filter: brightness(1.1);
    }
    .bm-btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .bm-btn-secondary {
      background: #f3f4f6;
      color: #374151;
    }
    .bm-btn-secondary:hover {
      background: #e5e7eb;
    }
    .bm-modal.dark .bm-btn-secondary {
      background: #374151;
      color: white;
    }
    .bm-btn-secondary:hover {
      background: #4b5563;
    }

    .bm-preview {
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 16px;
    }
    .bm-preview img {
      width: 100%;
      height: auto;
      display: block;
    }
    .bm-preview-overlay {
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .bm-loader {
      width: 40px;
      height: 40px;
      border: 3px solid rgba(255, 255, 255, 0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: bm-spin 1s linear infinite;
    }
    @keyframes bm-spin {
      to { transform: rotate(360deg); }
    }

    .bm-result {
      text-align: center;
    }
    .bm-size-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 80px;
      height: 80px;
      border-radius: 50%;
      background: var(--bm-primary);
      color: white;
      font-size: 32px;
      font-weight: 700;
      margin-bottom: 16px;
    }
    .bm-result-title {
      font-size: 18px;
      font-weight: 600;
      margin-bottom: 8px;
    }
    .bm-result-subtitle {
      color: #6b7280;
      margin-bottom: 24px;
    }

    .bm-measurements {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      margin-bottom: 24px;
    }
    .bm-measurement-card {
      background: #f9fafb;
      border-radius: 10px;
      padding: 12px;
      text-align: center;
    }
    .bm-modal.dark .bm-measurement-card {
      background: #374151;
    }
    .bm-measurement-label {
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 4px;
    }
    .bm-measurement-value {
      font-size: 18px;
      font-weight: 600;
    }

    .bm-fit-quality {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 16px;
    }
    .bm-fit-quality.perfect {
      background: #dcfce7;
      color: #166534;
    }
    .bm-fit-quality.good {
      background: #dbeafe;
      color: #1e40af;
    }
    .bm-fit-quality.acceptable {
      background: #fef3c7;
      color: #92400e;
    }
    .bm-fit-quality.poor {
      background: #fee2e2;
      color: #991b1b;
    }

    .bm-alternatives {
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid #e5e7eb;
    }
    .bm-modal.dark .bm-alternatives {
      border-color: #374151;
    }
    .bm-alternatives-title {
      font-size: 12px;
      color: #6b7280;
      margin-bottom: 8px;
    }
    .bm-alt-sizes {
      display: flex;
      justify-content: center;
      gap: 8px;
    }
    .bm-alt-size {
      padding: 6px 12px;
      border-radius: 6px;
      background: #f3f4f6;
      font-size: 14px;
      font-weight: 500;
    }
    .bm-modal.dark .bm-alt-size {
      background: #374151;
    }

    .bm-error {
      padding: 16px;
      background: #fee2e2;
      border-radius: 10px;
      color: #991b1b;
      text-align: center;
      margin-bottom: 16px;
    }

    .bm-camera-container {
      position: relative;
      border-radius: 12px;
      overflow: hidden;
      background: #000;
    }
    .bm-camera-video {
      width: 100%;
      display: block;
    }
    .bm-camera-guide {
      position: absolute;
      inset: 10%;
      border: 2px dashed rgba(255, 255, 255, 0.5);
      border-radius: 12px;
      pointer-events: none;
    }
    .bm-camera-controls {
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      display: flex;
      gap: 12px;
    }
    .bm-capture-btn {
      width: 64px;
      height: 64px;
      border-radius: 50%;
      background: white;
      border: 4px solid rgba(255, 255, 255, 0.3);
      cursor: pointer;
      transition: transform 0.2s;
    }
    .bm-capture-btn:hover {
      transform: scale(1.1);
    }
    .bm-capture-btn:active {
      transform: scale(0.95);
    }

    .bm-tabs {
      display: flex;
      gap: 4px;
      padding: 4px;
      background: #f3f4f6;
      border-radius: 10px;
      margin-bottom: 16px;
    }
    .bm-modal.dark .bm-tabs {
      background: #374151;
    }
    .bm-tab {
      flex: 1;
      padding: 10px;
      border: none;
      background: transparent;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      color: #6b7280;
      transition: all 0.2s;
    }
    .bm-tab.active {
      background: white;
      color: #111827;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    .bm-modal.dark .bm-tab.active {
      background: #1f2937;
      color: white;
    }

    .bm-fit-preference {
      margin-bottom: 16px;
    }
    .bm-fit-preference label {
      display: block;
      font-size: 14px;
      font-weight: 500;
      margin-bottom: 8px;
    }
    .bm-fit-options {
      display: flex;
      gap: 8px;
    }
    .bm-fit-option {
      flex: 1;
      padding: 10px;
      border: 2px solid #e5e7eb;
      border-radius: 8px;
      background: transparent;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.2s;
    }
    .bm-fit-option:hover {
      border-color: var(--bm-primary);
    }
    .bm-fit-option.active {
      border-color: var(--bm-primary);
      background: rgba(99, 102, 241, 0.1);
      color: var(--bm-primary);
    }
    .bm-modal.dark .bm-fit-option {
      border-color: #4b5563;
      color: white;
    }

    .bm-powered-by {
      text-align: center;
      font-size: 11px;
      color: #9ca3af;
      margin-top: 16px;
    }
    .bm-powered-by a {
      color: var(--bm-primary);
      text-decoration: none;
    }
  `;

  // Icons
  const ICONS = {
    ruler: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>',
    close: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
    camera: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>',
  };

  // Widget class
  class BodyMeasurementWidget {
    constructor(config) {
      this.config = { ...DEFAULT_CONFIG, ...config };
      this.state = {
        isOpen: false,
        step: 'upload', // upload, camera, processing, result
        selectedFile: null,
        previewUrl: null,
        measurements: null,
        recommendation: null,
        error: null,
        fitPreference: 'regular',
        inputMode: 'upload', // upload or camera
      };

      this.init();
    }

    init() {
      this.injectStyles();
      this.createButton();
      this.createModal();
      this.bindEvents();
    }

    injectStyles() {
      const style = document.createElement('style');
      style.id = 'bm-widget-styles';
      style.textContent = STYLES;
      document.head.appendChild(style);

      // Add CSS variable for primary color
      document.documentElement.style.setProperty('--bm-primary', this.config.primaryColor);
    }

    createButton() {
      this.button = document.createElement('button');
      this.button.className = `bm-widget-button ${this.config.position}`;
      this.button.style.backgroundColor = this.config.primaryColor;
      this.button.style.color = 'white';
      this.button.innerHTML = `${ICONS.ruler}<span>${this.config.buttonText}</span>`;
      document.body.appendChild(this.button);
    }

    createModal() {
      this.overlay = document.createElement('div');
      this.overlay.className = 'bm-modal-overlay';
      this.overlay.innerHTML = `
        <div class="bm-modal ${this.config.theme}">
          <div class="bm-modal-header">
            <h2>Find Your Perfect Size</h2>
            <button class="bm-close-btn">${ICONS.close}</button>
          </div>
          <div class="bm-modal-content">
            <!-- Upload Step -->
            <div class="bm-step active" data-step="upload">
              <div class="bm-tabs">
                <button class="bm-tab active" data-mode="upload">${ICONS.upload} Upload</button>
                <button class="bm-tab" data-mode="camera">${ICONS.camera} Camera</button>
              </div>

              <div class="bm-input-mode" data-mode="upload">
                <div class="bm-upload-area">
                  <div class="bm-upload-icon">${ICONS.upload}</div>
                  <p class="bm-upload-text">
                    <strong>Click to upload</strong> or drag and drop<br>
                    a full-body photo
                  </p>
                  <input type="file" accept="image/*" hidden>
                </div>
              </div>

              <div class="bm-input-mode" data-mode="camera" style="display: none;">
                <div class="bm-camera-container">
                  <video class="bm-camera-video" autoplay playsinline></video>
                  <div class="bm-camera-guide"></div>
                  <div class="bm-camera-controls">
                    <button class="bm-capture-btn"></button>
                  </div>
                </div>
              </div>
            </div>

            <!-- Preview Step -->
            <div class="bm-step" data-step="preview">
              <div class="bm-preview">
                <img src="" alt="Preview">
              </div>
              <div class="bm-fit-preference">
                <label>Fit Preference</label>
                <div class="bm-fit-options">
                  <button class="bm-fit-option" data-fit="tight">Tight</button>
                  <button class="bm-fit-option active" data-fit="regular">Regular</button>
                  <button class="bm-fit-option" data-fit="loose">Loose</button>
                </div>
              </div>
              <div style="display: flex; gap: 12px;">
                <button class="bm-btn bm-btn-secondary bm-back-btn">Back</button>
                <button class="bm-btn bm-btn-primary bm-process-btn" style="flex: 1;">Get My Size</button>
              </div>
            </div>

            <!-- Processing Step -->
            <div class="bm-step" data-step="processing">
              <div style="text-align: center; padding: 40px;">
                <div class="bm-loader" style="margin: 0 auto 16px; border-color: ${this.config.primaryColor}33; border-top-color: ${this.config.primaryColor};"></div>
                <p style="color: #6b7280;">Analyzing your measurements...</p>
              </div>
            </div>

            <!-- Result Step -->
            <div class="bm-step" data-step="result">
              <div class="bm-result">
                <div class="bm-size-badge"></div>
                <div class="bm-result-title">Your Recommended Size</div>
                <div class="bm-fit-quality"></div>
                <div class="bm-result-subtitle"></div>
                <div class="bm-measurements"></div>
                <div class="bm-alternatives"></div>
                <button class="bm-btn bm-btn-primary bm-try-again-btn" style="width: 100%;">Try Another Photo</button>
              </div>
            </div>

            <!-- Error Step -->
            <div class="bm-step" data-step="error">
              <div class="bm-error"></div>
              <button class="bm-btn bm-btn-primary bm-try-again-btn" style="width: 100%;">Try Again</button>
            </div>

            <div class="bm-powered-by">
              Powered by <a href="https://fitwhisperer.io" target="_blank">FitWhisperer</a>
            </div>
          </div>
        </div>
      `;
      document.body.appendChild(this.overlay);

      // Cache elements
      this.modal = this.overlay.querySelector('.bm-modal');
      this.fileInput = this.overlay.querySelector('input[type="file"]');
      this.uploadArea = this.overlay.querySelector('.bm-upload-area');
      this.previewImg = this.overlay.querySelector('.bm-preview img');
      this.video = this.overlay.querySelector('.bm-camera-video');
    }

    bindEvents() {
      // Button click
      this.button.addEventListener('click', () => this.open());

      // Close modal
      this.overlay.addEventListener('click', (e) => {
        if (e.target === this.overlay) this.close();
      });
      this.overlay.querySelector('.bm-close-btn').addEventListener('click', () => this.close());

      // Tab switching
      this.overlay.querySelectorAll('.bm-tab').forEach(tab => {
        tab.addEventListener('click', () => this.switchMode(tab.dataset.mode));
      });

      // File upload
      this.uploadArea.addEventListener('click', () => this.fileInput.click());
      this.fileInput.addEventListener('change', (e) => this.handleFile(e.target.files[0]));

      // Drag and drop
      this.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        this.uploadArea.classList.add('dragover');
      });
      this.uploadArea.addEventListener('dragleave', () => {
        this.uploadArea.classList.remove('dragover');
      });
      this.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        this.uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files[0]) {
          this.handleFile(e.dataTransfer.files[0]);
        }
      });

      // Camera capture
      this.overlay.querySelector('.bm-capture-btn').addEventListener('click', () => this.capturePhoto());

      // Fit preference
      this.overlay.querySelectorAll('.bm-fit-option').forEach(btn => {
        btn.addEventListener('click', () => {
          this.overlay.querySelectorAll('.bm-fit-option').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          this.state.fitPreference = btn.dataset.fit;
        });
      });

      // Back button
      this.overlay.querySelector('.bm-back-btn').addEventListener('click', () => this.goToStep('upload'));

      // Process button
      this.overlay.querySelector('.bm-process-btn').addEventListener('click', () => this.processImage());

      // Try again buttons
      this.overlay.querySelectorAll('.bm-try-again-btn').forEach(btn => {
        btn.addEventListener('click', () => this.reset());
      });
    }

    open() {
      this.state.isOpen = true;
      this.overlay.classList.add('open');
      document.body.style.overflow = 'hidden';
    }

    close() {
      this.state.isOpen = false;
      this.overlay.classList.remove('open');
      document.body.style.overflow = '';
      this.stopCamera();
    }

    switchMode(mode) {
      this.state.inputMode = mode;
      this.overlay.querySelectorAll('.bm-tab').forEach(t => t.classList.remove('active'));
      this.overlay.querySelector(`.bm-tab[data-mode="${mode}"]`).classList.add('active');
      this.overlay.querySelectorAll('.bm-input-mode').forEach(el => el.style.display = 'none');
      this.overlay.querySelector(`.bm-input-mode[data-mode="${mode}"]`).style.display = 'block';

      if (mode === 'camera') {
        this.startCamera();
      } else {
        this.stopCamera();
      }
    }

    async startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: 1280, height: 720 }
        });
        this.video.srcObject = stream;
        this.stream = stream;
      } catch (err) {
        console.error('Camera error:', err);
        this.showError('Could not access camera. Please allow camera permissions or upload an image.');
      }
    }

    stopCamera() {
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }
    }

    capturePhoto() {
      const canvas = document.createElement('canvas');
      canvas.width = this.video.videoWidth;
      canvas.height = this.video.videoHeight;
      canvas.getContext('2d').drawImage(this.video, 0, 0);

      canvas.toBlob(blob => {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        this.handleFile(file);
      }, 'image/jpeg', 0.8);
    }

    handleFile(file) {
      if (!file || !file.type.startsWith('image/')) {
        this.showError('Please select a valid image file.');
        return;
      }

      this.state.selectedFile = file;
      this.state.previewUrl = URL.createObjectURL(file);
      this.previewImg.src = this.state.previewUrl;
      this.stopCamera();
      this.goToStep('preview');
    }

    goToStep(step) {
      this.state.step = step;
      this.overlay.querySelectorAll('.bm-step').forEach(el => el.classList.remove('active'));
      this.overlay.querySelector(`.bm-step[data-step="${step}"]`).classList.add('active');
    }

    async processImage() {
      if (!this.state.selectedFile) return;

      this.goToStep('processing');

      try {
        // Process measurements
        const formData = new FormData();
        formData.append('file', this.state.selectedFile);

        const measurementResponse = await fetch(
          `${this.config.apiUrl}/api/v1/measurements/process-multi?api_key=${this.config.apiKey}`,
          { method: 'POST', body: formData }
        );

        if (!measurementResponse.ok) {
          throw new Error('Failed to process image');
        }

        const measurementData = await measurementResponse.json();
        this.state.measurements = measurementData;

        if (this.config.onMeasurement) {
          this.config.onMeasurement(measurementData);
        }

        // Get valid person
        const person = measurementData.measurements.find(p => p.is_valid);
        if (!person) {
          throw new Error('Could not detect a valid body in the image. Please try another photo.');
        }

        // Get size recommendation if product ID is provided
        if (this.config.productId) {
          const sizeResponse = await fetch(
            `${this.config.apiUrl}/api/v1/brands/products/${this.config.productId}/recommend-size?api_key=${this.config.apiKey}`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                product_id: this.config.productId,
                chest_circumference: person.chest_circumference,
                waist_circumference: person.waist_circumference,
                hip_circumference: person.hip_circumference,
                height: person.estimated_height_cm,
                fit_preference: this.state.fitPreference,
              }),
            }
          );

          if (sizeResponse.ok) {
            this.state.recommendation = await sizeResponse.json();
            if (this.config.onSizeRecommendation) {
              this.config.onSizeRecommendation(this.state.recommendation);
            }
          }
        }

        this.showResult(person);
      } catch (err) {
        console.error('Processing error:', err);
        this.showError(err.message || 'Failed to process image. Please try again.');
        if (this.config.onError) {
          this.config.onError(err);
        }
      }
    }

    showResult(person) {
      const rec = this.state.recommendation;
      const size = rec?.recommended_size || person.recommended_size || 'M';
      const fitQuality = rec?.fit_quality || 'good';
      const confidence = rec?.confidence || 0.85;

      // Update size badge
      this.overlay.querySelector('.bm-size-badge').textContent = size;

      // Update fit quality badge
      const fitQualityEl = this.overlay.querySelector('.bm-fit-quality');
      fitQualityEl.className = `bm-fit-quality ${fitQuality}`;
      fitQualityEl.innerHTML = `${ICONS.check} ${fitQuality.charAt(0).toUpperCase() + fitQuality.slice(1)} Fit`;

      // Update subtitle
      this.overlay.querySelector('.bm-result-subtitle').textContent =
        this.config.showConfidence ? `${Math.round(confidence * 100)}% confidence` : '';

      // Update measurements
      const measurementsHtml = [
        { label: 'Chest', value: person.chest_circumference },
        { label: 'Waist', value: person.waist_circumference },
        { label: 'Hip', value: person.hip_circumference },
        { label: 'Height', value: person.estimated_height_cm },
      ]
        .filter(m => m.value)
        .map(m => `
          <div class="bm-measurement-card">
            <div class="bm-measurement-label">${m.label}</div>
            <div class="bm-measurement-value">${m.value.toFixed(1)} cm</div>
          </div>
        `).join('');

      this.overlay.querySelector('.bm-measurements').innerHTML = measurementsHtml;

      // Update alternatives
      const altEl = this.overlay.querySelector('.bm-alternatives');
      if (rec?.alternative_sizes?.length) {
        altEl.innerHTML = `
          <div class="bm-alternatives-title">Also consider:</div>
          <div class="bm-alt-sizes">
            ${rec.alternative_sizes.map(s => `<span class="bm-alt-size">${s}</span>`).join('')}
          </div>
        `;
        altEl.style.display = 'block';
      } else {
        altEl.style.display = 'none';
      }

      this.goToStep('result');
    }

    showError(message) {
      this.overlay.querySelector('.bm-error').textContent = message;
      this.goToStep('error');
    }

    reset() {
      this.state.selectedFile = null;
      this.state.previewUrl = null;
      this.state.measurements = null;
      this.state.recommendation = null;
      this.state.error = null;
      this.fileInput.value = '';
      this.goToStep('upload');
    }

    // Public methods
    destroy() {
      this.close();
      this.button.remove();
      this.overlay.remove();
      document.getElementById('bm-widget-styles')?.remove();
    }

    setProductId(productId) {
      this.config.productId = productId;
    }
  }

  // Expose to global
  window.BodyMeasurementWidget = BodyMeasurementWidget;

  // Auto-init if data attribute is present
  document.addEventListener('DOMContentLoaded', () => {
    const script = document.querySelector('script[data-bm-api-key]');
    if (script) {
      new BodyMeasurementWidget({
        apiKey: script.dataset.bmApiKey,
        productId: script.dataset.bmProductId,
        theme: script.dataset.bmTheme,
        primaryColor: script.dataset.bmColor,
        position: script.dataset.bmPosition,
        buttonText: script.dataset.bmButtonText,
      });
    }
  });

})(window, document);
