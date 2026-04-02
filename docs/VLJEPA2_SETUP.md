# VL-JEPA2 Setup Guide for 16GB RAM Systems

**Model:** `facebook/vjepa2-vitl-fpc64-256`

This guide shows you how to run VL-JEPA2 for behavior analysis on a system with:
- 16GB System RAM
- 8GB VRAM (Intel UHD Graphics)
- Windows 11

---

## Quick Start

### Step 1: Install VL-JEPA2 Dependencies

```bash
# Activate your virtual environment
.venv\Scripts\activate

# Install PyTorch (with Intel GPU support if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install transformers and dependencies
pip install transformers>=4.35.0 accelerate>=0.24.0

# Install quantization support (for 8-bit inference)
pip install bitsandbytes>=0.41.0

# Install OpenVINO for Intel GPU acceleration
pip install optimum-intel openvino-dev
```

### Step 2: Configure for VL-JEPA2

Edit `configs/default.yaml`:

```yaml
behavior:
  backend: "vljepa2"
  model_name: "facebook/vjepa2-vitl-fpc64-256"
  use_quantization: true      # IMPORTANT for 16GB RAM
  use_openvino: true          # Enable for Intel UHD Graphics
  input_resolution: 224       # 224 for speed, 256 for accuracy
  confidence_threshold: 0.6
```

### Step 3: First Run (Model Download)

```bash
python main.py --config configs/default.yaml
```

On first run, VL-JEPA2 will download (~1-2GB). Subsequent runs use cached model.

---

## Performance Optimization for 16GB RAM

### Option 1: CPU-Only Mode (Most Compatible)

```yaml
behavior:
  backend: "vljepa2"
  use_quantization: true
  use_openvino: false
  input_resolution: 224
```

**Expected Performance:**
- FPS: 3-5
- Memory: 6-8GB RAM
- Latency: 200-300ms

### Option 2: OpenVINO GPU Mode (Intel UHD Graphics)

```yaml
behavior:
  backend: "vljepa2"
  use_quantization: true
  use_openvino: true
  input_resolution: 224
```

**Expected Performance:**
- FPS: 5-8
- Memory: 4-6GB RAM + shared GPU memory
- Latency: 120-200ms

**Setup for OpenVINO:**

```bash
# Install OpenVINO
pip install openvino-dev optimum-intel

# Verify installation
python -c "from openvino.runtime import Core; print(Core().available_devices)"
```

### Option 3: Hybrid Mode (Recommended)

Use VL-JEPA2 for key frames, MediaPipe for others:

```yaml
behavior:
  backend: "hybrid"
  vljepa2_keyframe_interval: 10  # Run VL-JEPA2 every 10 frames
  use_quantization: true
  input_resolution: 224
```

**Expected Performance:**
- FPS: 8-12 (effective)
- Memory: 4-6GB RAM
- Best balance of accuracy and speed

---

## Memory Optimization Techniques

### 1. Enable 8-bit Quantization

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
```

Reduces model memory from ~4GB to ~2GB.

### 2. Reduce Input Resolution

```yaml
behavior:
  input_resolution: 224  # Instead of 256 or higher
```

Smaller input = less memory + faster inference.

### 3. Increase Frame Skip

```yaml
optimization:
  frame_skip: 3  # Process every 4th frame with VL-JEPA2
```

### 4. Enable Garbage Collection

```python
# In main.py, add after processing:
import gc
if frame_count % 100 == 0:
    gc.collect()
    torch.cuda.empty_cache()  # If using GPU
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1:** Enable quantization
```yaml
behavior:
  use_quantization: true
```

**Solution 2:** Reduce resolution
```yaml
behavior:
  input_resolution: 224  # or even 192
```

**Solution 3:** Switch to CPU
```yaml
behavior:
  use_openvino: false
```

### Issue: "Model download failed"

**Solution:** Manual download
```bash
# Download model manually
huggingface-cli download facebook/vjepa2-vitl-fpc64-256 --local-dir models/vljepa2

# Update config
behavior:
  model_name: "./models/vljepa2"
```

### Issue: "OpenVINO not working"

**Solution:** Reinstall OpenVINO
```bash
pip uninstall openvino openvino-dev optimum-intel
pip install openvino-dev optimum-intel

# Verify
python -c "from optimum.intel import OVModelForImageClassification; print('OK')"
```

### Issue: Very slow inference (<2 FPS)

**Solutions:**
1. Use hybrid mode (see Option 3 above)
2. Reduce `frame_skip` in config
3. Fall back to MediaPipe for most frames

---

## Benchmarking Your System

Run the benchmark script:

```bash
python benchmarks/vljepa2_benchmark.py
```

Expected results for 16GB RAM + Intel UHD:

| Configuration | FPS | Memory | Recommended |
|--------------|-----|--------|-------------|
| CPU, quantized, 224px | 3-5 | 6GB | Testing only |
| OpenVINO GPU, quantized, 224px | 5-8 | 4-6GB | Production |
| Hybrid mode | 8-12 | 4GB | Best balance |
| MediaPipe only | 15-20 | 2GB | Max performance |

---

## Full Configuration Examples

### Example 1: Maximum Performance (Lowest Memory)

```yaml
behavior:
  backend: "mediapipe"  # Skip VL-JEPA2 entirely

optimization:
  frame_skip: 1
  max_cameras: 1
  cpu_workers: 2
```

### Example 2: Balanced (Recommended for Most)

```yaml
behavior:
  backend: "hybrid"
  vljepa2_keyframe_interval: 10
  use_quantization: true
  input_resolution: 224

optimization:
  frame_skip: 2
  max_cameras: 1
  cpu_workers: 1
  gc_interval: 100
```

### Example 3: Maximum Accuracy (Higher Memory)

```yaml
behavior:
  backend: "vljepa2"
  use_quantization: false  # Full precision
  use_openvino: true
  input_resolution: 256

optimization:
  frame_skip: 1
  max_cameras: 1
  cpu_workers: 1
```

---

## API Usage

### Basic Usage

```python
from src.detection.vljepa2 import create_vljepa2_analyzer

# Create analyzer
analyzer = create_vljepa2_analyzer(
    use_quantization=True,
    use_openvino=True,
    input_resolution=224
)

# Load model
analyzer.load_model()

# Analyze frame
behaviors = analyzer.analyze(frame, detections, tracks)

for behavior in behaviors:
    print(f"Detected: {behavior['behavior']} ({behavior['confidence']:.2f})")

# Cleanup
analyzer.release()
```

### Hybrid Mode

```python
from src.detection.behavior_vjepa import BehaviorAnalyzer, BehaviorBackend

# Create hybrid analyzer
analyzer = BehaviorAnalyzer(
    backend=BehaviorBackend.MEDIAPIPE,  # Primary
    use_vjepa2=False  # Enable VL-JEPA2 for keyframes
)

# Process every frame with MediaPipe, every 10th with VL-JEPA2
for i, frame in enumerate(frames):
    if i % 10 == 0:
        # Use VL-JEPA2 on keyframes
        behaviors = analyzer.analyze_vljepa2(frame)
    else:
        # Use MediaPipe on other frames
        behaviors = analyzer.analyze(frame)
```

---

## Model Information

**facebook/vjepa2-vitl-fpc64-256**

- **Architecture:** Vision Transformer (ViT-Large)
- **Input Size:** 256x256 (default), 224x224 (optimized)
- **Parameters:** ~300M
- **Memory (Full Precision):** ~4GB
- **Memory (8-bit Quantized):** ~2GB
- **License:** Apache 2.0

**HuggingFace:** https://huggingface.co/facebook/vjepa2-vitl-fpc64-256

**Paper:** "Self-Supervised Learning from Images with a Joint Embedding Predictive Architecture" - Meta AI

---

## Next Steps

1. Start with hybrid mode for best balance
2. Benchmark on your specific hardware
3. Adjust `frame_skip` and `input_resolution` based on FPS
4. Monitor memory usage with Task Manager
5. If OOM, enable quantization or reduce resolution

For more help, see the main [SETUP.md](../SETUP.md).
