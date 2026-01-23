'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import { Loader2, RotateCcw, ZoomIn, ZoomOut, Download, User } from 'lucide-react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

// Types
interface MeshData {
  vertices: number[][];
  faces: number[][];
  vertex_colors?: number[][];
  uv_coordinates?: number[][];
  measurements: {
    height_cm: number;
    chest_cm: number;
    waist_cm: number;
    hip_cm: number;
    shoulder_cm: number;
    arm_length_cm: number;
    inseam_cm: number;
    gender: string;
  };
}

// View mode types
type ViewMode = 'mannequin' | 'texture' | 'depth2.5d' | 'pifuhd';

interface Body3DViewerInnerProps {
  heightCm?: number;
  gender?: 'male' | 'female' | 'neutral';
  weightFactor?: number;
  apiBaseUrl?: string;
  showControls?: boolean;
  showMeasurements?: boolean;
  autoRotate?: boolean;
  className?: string;
  imageUrl?: string; // URL of the uploaded image for texture mapping
  viewMode?: ViewMode; // Which 3D visualization mode to use
  onViewModeChange?: (mode: ViewMode) => void; // Callback when mode changes
}

export default function Body3DViewerInner({
  heightCm = 175,
  gender = 'neutral',
  weightFactor = 0,
  apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  showControls = true,
  showMeasurements = true,
  autoRotate = true,
  className = '',
  imageUrl,
  viewMode = 'mannequin',
  onViewModeChange,
}: Body3DViewerInnerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);
  const meshRef = useRef<THREE.Mesh | null>(null);
  const textureRef = useRef<THREE.Texture | null>(null);
  const animationRef = useRef<number | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [meshData, setMeshData] = useState<MeshData | null>(null);
  const [isAutoRotating, setIsAutoRotating] = useState(autoRotate);
  const [sceneReady, setSceneReady] = useState(false);
  const [textureLoaded, setTextureLoaded] = useState(false);
  const [depthData, setDepthData] = useState<number[][] | null>(null);
  const depth2DMeshRef = useRef<THREE.Mesh | null>(null);

  // Load texture from image URL (works with blob URLs)
  const loadTexture = useCallback(async (url: string): Promise<THREE.Texture | null> => {
    return new Promise((resolve) => {
      // Create an image element to load the blob URL
      const img = new Image();
      img.crossOrigin = 'anonymous';

      img.onload = () => {
        // Create texture from the loaded image
        const texture = new THREE.Texture(img);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.needsUpdate = true;
        texture.flipY = true; // Flip Y for correct orientation
        texture.wrapS = THREE.ClampToEdgeWrapping;
        texture.wrapT = THREE.ClampToEdgeWrapping;
        console.log('Texture loaded successfully:', img.width, 'x', img.height);
        resolve(texture);
      };

      img.onerror = (err) => {
        console.error('Error loading texture image:', err);
        resolve(null);
      };

      img.src = url;
    });
  }, []);

  // Convert image URL to base64
  const imageUrlToBase64 = useCallback(async (url: string): Promise<string> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }
        ctx.drawImage(img, 0, 0);
        const base64 = canvas.toDataURL('image/jpeg', 0.9);
        resolve(base64);
      };
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = url;
    });
  }, []);

  // Fetch depth map from API
  const fetchDepthMap = useCallback(async (imageUrl: string): Promise<number[][] | null> => {
    try {
      console.log('Fetching depth map...');
      const base64Image = await imageUrlToBase64(imageUrl);

      const response = await fetch(`${apiBaseUrl}/api/v1/depth/depth/estimate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64Image,
          resolution: 256, // Lower resolution for performance
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch depth: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Depth map received:', data.width, 'x', data.height);
      return data.depth_map;
    } catch (err) {
      console.error('Error fetching depth map:', err);
      return null;
    }
  }, [apiBaseUrl, imageUrlToBase64]);

  // Create 2.5D mesh from image and depth map
  const create2_5DMesh = useCallback(() => {
    if (!sceneRef.current || !textureRef.current || !depthData) return;

    console.log('Creating 2.5D mesh...');

    // Remove existing 2.5D mesh
    if (depth2DMeshRef.current) {
      sceneRef.current.remove(depth2DMeshRef.current);
      depth2DMeshRef.current.geometry.dispose();
      if (depth2DMeshRef.current.material instanceof THREE.Material) {
        depth2DMeshRef.current.material.dispose();
      }
    }

    // Also remove the regular mesh if it exists
    if (meshRef.current) {
      sceneRef.current.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      if (meshRef.current.material instanceof THREE.Material) {
        meshRef.current.material.dispose();
      }
      meshRef.current = null;
    }

    const depthHeight = depthData.length;
    const depthWidth = depthData[0].length;

    // Create a plane geometry with segments matching depth map resolution
    const segmentsX = Math.min(depthWidth, 128); // Limit segments for performance
    const segmentsY = Math.min(depthHeight, 128);

    // Aspect ratio based on depth map
    const aspectRatio = depthWidth / depthHeight;
    const planeWidth = 2 * aspectRatio;
    const planeHeight = 2;

    const geometry = new THREE.PlaneGeometry(planeWidth, planeHeight, segmentsX, segmentsY);
    const positions = geometry.attributes.position;

    // Displace vertices based on depth
    const depthScale = 0.5; // How much depth affects the displacement

    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);

      // Map vertex position to depth map coordinates
      const u = (x / planeWidth + 0.5);
      const v = (y / planeHeight + 0.5);

      const depthX = Math.floor(u * (depthWidth - 1));
      const depthY = Math.floor(v * (depthHeight - 1)); // Match texture orientation

      // Get depth value (clamped to valid range)
      const clampedX = Math.max(0, Math.min(depthWidth - 1, depthX));
      const clampedY = Math.max(0, Math.min(depthHeight - 1, depthY));
      const depth = depthData[clampedY][clampedX];

      // Displace Z based on depth (closer = more positive Z)
      positions.setZ(i, depth * depthScale);
    }

    geometry.computeVertexNormals();

    // Create material with the texture
    const material = new THREE.MeshStandardMaterial({
      map: textureRef.current,
      side: THREE.DoubleSide,
      roughness: 0.8,
      metalness: 0.1,
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(0, 1, 0); // Center in view

    sceneRef.current.add(mesh);
    depth2DMeshRef.current = mesh;

    console.log('2.5D mesh created successfully');
  }, [depthData]);

  // Fetch mesh data from API
  const fetchMesh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Use textured endpoint if we have an image
      const endpoint = imageUrl
        ? `${apiBaseUrl}/api/v1/mesh3d/mesh3d/generate-textured`
        : `${apiBaseUrl}/api/v1/mesh3d/mesh3d/generate`;

      let response;

      if (imageUrl) {
        // POST request for textured mesh
        response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            height_cm: heightCm,
            gender: gender,
            weight_factor: weightFactor,
          }),
        });
      } else {
        // GET request for regular mesh
        response = await fetch(`${endpoint}?height_cm=${heightCm}&gender=${gender}&weight_factor=${weightFactor}`);
      }

      if (!response.ok) {
        throw new Error(`Failed to fetch mesh: ${response.statusText}`);
      }

      const data: MeshData = await response.json();
      setMeshData(data);

      // Load texture if we have an image URL
      if (imageUrl) {
        const texture = await loadTexture(imageUrl);
        if (texture) {
          textureRef.current = texture;
          setTextureLoaded(true);
        }

        // Also fetch depth map for depth2.5d mode
        if (viewMode === 'depth2.5d') {
          const depth = await fetchDepthMap(imageUrl);
          if (depth) {
            setDepthData(depth);
          }
        }
      }
    } catch (err) {
      console.error('Error fetching mesh:', err);
      setError(err instanceof Error ? err.message : 'Failed to load 3D model');
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, heightCm, gender, weightFactor, imageUrl, loadTexture, viewMode, fetchDepthMap]);

  // Initialize Three.js scene
  const initScene = useCallback(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
    camera.position.set(0, 0.8, 3);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.minDistance = 1;
    controls.maxDistance = 10;
    controls.target.set(0, 0.8, 0);
    controls.autoRotate = isAutoRotating;
    controls.autoRotateSpeed = 2;
    controlsRef.current = controls;

    // Lights - brighter for texture visibility
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);

    const frontLight = new THREE.DirectionalLight(0xffffff, 0.5);
    frontLight.position.set(0, 5, 10);
    scene.add(frontLight);

    const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
    backLight.position.set(-5, 5, -5);
    scene.add(backLight);

    // Grid helper
    const gridHelper = new THREE.GridHelper(4, 20, 0x444444, 0x333333);
    scene.add(gridHelper);

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);

      if (controlsRef.current) {
        controlsRef.current.update();
      }

      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth;
      const h = containerRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    setSceneReady(true);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [isAutoRotating]);

  // Create measurement band colors for mannequin mode
  const createMannequinColors = useCallback((vertices: number[][]): Float32Array => {
    const colors = new Float32Array(vertices.length * 3);

    // Find Y bounds for the body
    let minY = Infinity, maxY = -Infinity;
    for (const v of vertices) {
      if (v[1] < minY) minY = v[1];
      if (v[1] > maxY) maxY = v[1];
    }
    const bodyHeight = maxY - minY;

    // Measurement band positions (as fraction of height from bottom)
    const chestLevel = 0.72;
    const waistLevel = 0.62;
    const hipLevel = 0.52;
    const bandWidth = 0.04; // Width of each colored band

    // Skin tone colors (gradient for more natural look)
    const skinBase = [0.96, 0.82, 0.71]; // Light skin tone
    const skinDark = [0.88, 0.72, 0.61]; // Slightly darker for variation

    // Measurement band colors (vibrant but not too bright)
    const chestColor = [0.95, 0.30, 0.30]; // Red
    const waistColor = [0.30, 0.85, 0.40]; // Green
    const hipColor = [0.35, 0.50, 0.95];   // Blue

    for (let i = 0; i < vertices.length; i++) {
      const v = vertices[i];
      const yFrac = (v[1] - minY) / bodyHeight;

      let color = skinBase;

      // Add subtle variation based on position for more natural look
      const variation = Math.sin(v[0] * 10) * 0.02;

      // Check if vertex is in a measurement band
      if (Math.abs(yFrac - chestLevel) < bandWidth) {
        color = chestColor;
      } else if (Math.abs(yFrac - waistLevel) < bandWidth) {
        color = waistColor;
      } else if (Math.abs(yFrac - hipLevel) < bandWidth) {
        color = hipColor;
      } else {
        // Blend skin tones for natural look
        const blend = (Math.sin(yFrac * Math.PI * 2) + 1) / 2;
        color = [
          skinBase[0] * (1 - blend * 0.1) + skinDark[0] * (blend * 0.1) + variation,
          skinBase[1] * (1 - blend * 0.1) + skinDark[1] * (blend * 0.1) + variation,
          skinBase[2] * (1 - blend * 0.1) + skinDark[2] * (blend * 0.1) + variation,
        ];
      }

      colors[i * 3] = color[0];
      colors[i * 3 + 1] = color[1];
      colors[i * 3 + 2] = color[2];
    }

    return colors;
  }, []);

  // Create mesh from data
  const createMesh = useCallback(() => {
    if (!meshData || !sceneRef.current) return;

    console.log('Creating mesh with viewMode:', viewMode);

    // Remove existing mesh
    if (meshRef.current) {
      sceneRef.current.remove(meshRef.current);
      meshRef.current.geometry.dispose();
      if (meshRef.current.material instanceof THREE.Material) {
        meshRef.current.material.dispose();
      }
    }

    // Create geometry
    const geometry = new THREE.BufferGeometry();

    // Vertices
    const vertices = new Float32Array(meshData.vertices.flat());
    geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

    // Faces (indices)
    const indices = new Uint32Array(meshData.faces.flat());
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));

    // Compute normals for proper lighting
    geometry.computeVertexNormals();

    // Material based on view mode
    let material: THREE.Material;

    switch (viewMode) {
      case 'mannequin':
        // Clean mannequin with colored measurement bands
        const mannequinColors = createMannequinColors(meshData.vertices);
        geometry.setAttribute('color', new THREE.BufferAttribute(mannequinColors, 3));
        material = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.6,
          metalness: 0.05,
          side: THREE.DoubleSide,
        });
        break;

      case 'texture':
        // Textured mesh with uploaded image
        if (meshData.uv_coordinates && meshData.uv_coordinates.length > 0) {
          const uvs = new Float32Array(meshData.uv_coordinates.flat());
          geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        }
        if (textureRef.current && meshData.uv_coordinates) {
          material = new THREE.MeshBasicMaterial({
            map: textureRef.current,
            side: THREE.DoubleSide,
          });
        } else {
          // Fallback if no texture
          material = new THREE.MeshStandardMaterial({
            color: 0xf5c8a8,
            roughness: 0.7,
            metalness: 0.1,
            side: THREE.DoubleSide,
          });
        }
        break;

      case 'depth2.5d':
        // Depth 2.5D mode is handled by create2_5DMesh function
        // Show a loading indicator mesh while depth is being processed
        if (!depthData) {
          material = new THREE.MeshStandardMaterial({
            color: 0x6366f1, // Indigo to indicate depth mode loading
            roughness: 0.5,
            metalness: 0.2,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5,
          });
        } else {
          // Skip mesh creation here, create2_5DMesh will handle it
          return;
        }
        break;

      case 'pifuhd':
        // TODO: Implement PIFuHD full 3D reconstruction
        // For now, show mannequin with a different color to indicate it's WIP
        material = new THREE.MeshStandardMaterial({
          color: 0x10b981, // Emerald to indicate PIFuHD mode (WIP)
          roughness: 0.4,
          metalness: 0.3,
          side: THREE.DoubleSide,
          wireframe: false,
        });
        break;

      default:
        material = new THREE.MeshStandardMaterial({
          color: 0xf5c8a8,
          roughness: 0.7,
          metalness: 0.1,
          side: THREE.DoubleSide,
        });
    }

    // Create mesh
    const mesh = new THREE.Mesh(geometry, material);
    sceneRef.current.add(mesh);
    meshRef.current = mesh;
  }, [meshData, textureLoaded, viewMode, createMannequinColors]);

  // Initialize scene on mount
  useEffect(() => {
    const cleanup = initScene();
    fetchMesh();
    return cleanup;
  }, [initScene, fetchMesh]);

  // Update mesh when data changes
  useEffect(() => {
    if (meshData && sceneReady) {
      // For depth2.5d mode, we need depth data first
      if (viewMode === 'depth2.5d') {
        if (depthData && textureLoaded) {
          create2_5DMesh();
        }
      } else {
        createMesh();
      }
    }
  }, [meshData, sceneReady, createMesh, textureLoaded, viewMode, depthData, create2_5DMesh]);

  // Update auto-rotate
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isAutoRotating;
    }
  }, [isAutoRotating]);

  // Re-fetch when parameters change
  useEffect(() => {
    if (sceneReady) {
      // Reset depth data when mode changes
      if (viewMode !== 'depth2.5d') {
        setDepthData(null);
      }
      fetchMesh();
    }
  }, [heightCm, gender, weightFactor, imageUrl, sceneReady, fetchMesh, viewMode]);

  // Control handlers
  const handleResetView = () => {
    if (cameraRef.current && controlsRef.current) {
      cameraRef.current.position.set(0, 0.8, 3);
      controlsRef.current.target.set(0, 0.8, 0);
      controlsRef.current.update();
    }
  };

  const handleZoom = (delta: number) => {
    if (cameraRef.current) {
      const direction = new THREE.Vector3();
      cameraRef.current.getWorldDirection(direction);
      cameraRef.current.position.addScaledVector(direction, delta);
    }
  };

  const handleDownloadOBJ = async () => {
    const url = `${apiBaseUrl}/api/v1/mesh3d/mesh3d/export/obj?height_cm=${heightCm}&gender=${gender}&weight_factor=${weightFactor}`;
    window.open(url, '_blank');
  };

  return (
    <div className={`relative bg-gray-900 rounded-lg overflow-hidden ${className}`}>
      {/* 3D Canvas Container */}
      <div
        ref={containerRef}
        className="w-full h-[500px]"
        style={{ minHeight: '400px' }}
      />

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="text-center">
            <Loader2 className="w-12 h-12 animate-spin text-indigo-500 mx-auto mb-2" />
            <p className="text-gray-400">
              {imageUrl ? 'Loading textured 3D model...' : 'Loading 3D model...'}
            </p>
          </div>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
          <div className="text-center p-4 max-w-md">
            <p className="text-red-400 mb-4">{error}</p>
            <button
              onClick={fetchMesh}
              className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* View Mode Selector */}
      {!loading && !error && (
        <div className="absolute top-4 left-4 flex flex-col gap-2">
          <div className="bg-gray-800/90 rounded-lg p-2">
            <p className="text-xs text-gray-400 mb-2 font-medium">View Mode</p>
            <div className="flex flex-col gap-1">
              <button
                onClick={() => onViewModeChange?.('mannequin')}
                className={`px-3 py-1.5 text-xs rounded transition-colors ${
                  viewMode === 'mannequin'
                    ? 'bg-indigo-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                3D Mannequin
              </button>
              {imageUrl && (
                <button
                  onClick={() => onViewModeChange?.('texture')}
                  className={`px-3 py-1.5 text-xs rounded transition-colors ${
                    viewMode === 'texture'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Photo Texture
                </button>
              )}
              {imageUrl && (
                <button
                  onClick={() => onViewModeChange?.('depth2.5d')}
                  className={`px-3 py-1.5 text-xs rounded transition-colors ${
                    viewMode === 'depth2.5d'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  Depth 2.5D
                </button>
              )}
              {imageUrl && (
                <button
                  onClick={() => onViewModeChange?.('pifuhd')}
                  className={`px-3 py-1.5 text-xs rounded transition-colors ${
                    viewMode === 'pifuhd'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                  title="Requires GPU - not available in this environment"
                >
                  PIFuHD 3D
                  <span className="ml-1 text-[10px] text-gray-400">(GPU)</span>
                </button>
              )}
            </div>
          </div>

          {/* Mode indicator */}
          {viewMode === 'mannequin' && (
            <div className="bg-indigo-600/80 text-white text-xs px-2 py-1 rounded">
              Clean 3D Model with Measurements
            </div>
          )}
          {viewMode === 'texture' && textureLoaded && (
            <div className="bg-green-600/80 text-white text-xs px-2 py-1 rounded">
              Photo Texture Applied
            </div>
          )}
          {viewMode === 'depth2.5d' && !depthData && (
            <div className="bg-yellow-600/80 text-white text-xs px-2 py-1 rounded flex items-center gap-1">
              <Loader2 className="w-3 h-3 animate-spin" />
              Loading Depth...
            </div>
          )}
          {viewMode === 'depth2.5d' && depthData && (
            <div className="bg-cyan-600/80 text-white text-xs px-2 py-1 rounded">
              Depth 2.5D View Active
            </div>
          )}
          {viewMode === 'pifuhd' && (
            <div className="bg-gray-600/80 text-white text-xs px-2 py-1 rounded">
              PIFuHD 3D - Requires GPU
            </div>
          )}
        </div>
      )}

      {/* Controls */}
      {showControls && !loading && !error && (
        <div className="absolute bottom-4 left-4 flex gap-2">
          <button
            onClick={handleResetView}
            className="p-2 bg-gray-800/80 rounded-lg text-white hover:bg-gray-700"
            title="Reset View"
          >
            <RotateCcw className="w-5 h-5" />
          </button>
          <button
            onClick={() => handleZoom(0.5)}
            className="p-2 bg-gray-800/80 rounded-lg text-white hover:bg-gray-700"
            title="Zoom In"
          >
            <ZoomIn className="w-5 h-5" />
          </button>
          <button
            onClick={() => handleZoom(-0.5)}
            className="p-2 bg-gray-800/80 rounded-lg text-white hover:bg-gray-700"
            title="Zoom Out"
          >
            <ZoomOut className="w-5 h-5" />
          </button>
          <button
            onClick={() => setIsAutoRotating(!isAutoRotating)}
            className={`p-2 rounded-lg text-white ${
              isAutoRotating ? 'bg-indigo-600' : 'bg-gray-800/80 hover:bg-gray-700'
            }`}
            title="Toggle Auto-Rotate"
          >
            <RotateCcw className={`w-5 h-5 ${isAutoRotating ? 'animate-spin' : ''}`} style={{ animationDuration: '3s' }} />
          </button>
          <button
            onClick={handleDownloadOBJ}
            className="p-2 bg-gray-800/80 rounded-lg text-white hover:bg-gray-700"
            title="Download OBJ"
          >
            <Download className="w-5 h-5" />
          </button>
        </div>
      )}

      {/* Measurements panel */}
      {showMeasurements && meshData && !loading && (
        <div className="absolute top-4 right-4 bg-gray-800/90 rounded-lg p-4 text-white text-sm">
          <div className="flex items-center gap-2 mb-3 pb-2 border-b border-gray-700">
            <User className="w-5 h-5 text-indigo-400" />
            <span className="font-semibold">Body Measurements</span>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Height:</span>
              <span className="font-mono">{meshData.measurements.height_cm} cm</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Chest:</span>
              <span className="font-mono text-red-400">{meshData.measurements.chest_cm} cm</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Waist:</span>
              <span className="font-mono text-green-400">{meshData.measurements.waist_cm} cm</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Hip:</span>
              <span className="font-mono text-blue-400">{meshData.measurements.hip_cm} cm</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Shoulder:</span>
              <span className="font-mono">{meshData.measurements.shoulder_cm} cm</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Arm:</span>
              <span className="font-mono">{meshData.measurements.arm_length_cm} cm</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Inseam:</span>
              <span className="font-mono">{meshData.measurements.inseam_cm} cm</span>
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-gray-700 text-xs text-gray-500">
            Gender: {meshData.measurements.gender}
          </div>
        </div>
      )}

      {/* Instructions */}
      {!loading && !error && (
        <div className="absolute bottom-4 right-4 text-gray-500 text-xs">
          Drag to rotate • Scroll to zoom • Right-click to pan
        </div>
      )}
    </div>
  );
}
