# Behavior Analysis Backend Options

## Overview

This document explains the behavior analysis options for the AI Surveillance system, including feasibility for different hardware configurations.

## Available Backends

### 1. MediaPipe Pose (Recommended for 16GB RAM)

**Memory Usage:** ~500MB
**Latency:** 15-30ms per frame
**Accuracy:** Good for basic behaviors

**Supported Behaviors:**
- Raised arms detection
- Fighting stance
- Running (via trajectory)
- Loitering (via trajectory)

**Pros:**
- Very fast on CPU
- Low memory footprint
- Works on Intel UHD Graphics
- No GPU required

**Cons:**
- Limited to pose-based behaviors
- Less contextual understanding

**Configuration:**
```python
from src.detection.behavior_vjepa import create_behavior_analyzer

# Recommended for 16GB RAM + Intel UHD
analyzer = create_behavior_analyzer(system_ram_gb=16, vram_gb=0)
```

### 2. VL-JEPA2 (Experimental, Requires 8GB+ VRAM)

**Memory Usage:** 4-8GB VRAM + 2GB system RAM
**Latency:** 100-200ms per frame
**Accuracy:** Excellent for complex behaviors

**About VL-JEPA2:**

VL-JEPA2 is Meta AI's Vision-Language Joint Embedding Predictive Architecture.
It's a self-supervised model that learns visual representations without
requiring labeled data.

**Key Papers & Resources:**
- [JEPA Paper (LeCun et al.)](https://ai.meta.com/research/joint-embedding-predictive-architecture/)
- [V-JEPA Details](https://ai.meta.com/blog/v-jepa-meta-ai-self-supervised-learning/)
- [Meta AI Blog](https://ai.meta.com/blog/)

**Supported Behaviors:**
- All MediaPipe behaviors plus:
- Contextual anomaly detection
- Group behavior analysis
- Object interaction understanding
- Temporal activity recognition

**Pros:**
- Much richer behavior understanding
- Context-aware predictions
- Can detect complex suspicious patterns

**Cons:**
- Requires 8GB+ dedicated VRAM
- Higher latency
- May need model quantization for real-time

**Configuration:**
```python
from src.detection.behavior_vjepa import create_behavior_analyzer

# Only for systems with 8GB+ VRAM
analyzer = create_behavior_analyzer(
    system_ram_gb=16,
    vram_gb=8,
    prefer_vjepa2=True
)
```

### 3. Rule-Based (Fallback)

**Memory Usage:** ~100MB
**Latency:** <5ms per frame
**Accuracy:** Basic but reliable

**Supported Behaviors:**
- Weapon detection (via YOLO classes)
- Crowd formation (people count)
- Running (via trajectory velocity)
- Loitering (via position history)

**Pros:**
- Works on any system
- Fastest option
- No additional dependencies

**Cons:**
- Limited behavior vocabulary
- No pose analysis

## Feasibility Analysis for Your System

### Your Specs: 16GB RAM, 8GB VRAM, Intel UHD Graphics

**Recommendation: Use MediaPipe + Rule-Based Hybrid**

Since you have Intel UHD Graphics (integrated) rather than dedicated 8GB VRAM,
VL-JEPA2 would need to use system RAM, which would:

1. Be significantly slower (10-20 FPS → 2-5 FPS)
2. Cause memory pressure on 16GB system
3. Not provide real-time performance

**Optimal Configuration:**

```yaml
# configs/default.yaml
behavior:
  backend: "mediapipe"      # Primary backend
  model_complexity: 0       # Lite model
  enable_segmentation: false

optimization:
  frame_skip: 3             # Process every 4th frame
  max_cameras: 1
```

### If You Have Dedicated 8GB VRAM (RTX 3060+)

Then VL-JEPA2 becomes feasible:

```yaml
behavior:
  backend: "vljepa2"
  use_gpu: true
  quantize: true            # Use INT8 quantization

optimization:
  frame_skip: 1             # Can process more frames
  max_cameras: 1
```

## Implementation in This Project

The `behavior_vjepa.py` module provides automatic backend selection:

```python
from src.detection.behavior_vjepa import create_behavior_analyzer

# Auto-select based on hardware
analyzer = create_behavior_analyzer(
    system_ram_gb=16,
    vram_gb=0,              # 0 for Intel UHD
    prefer_vjepa2=False     # Don't force VL-JEPA2
)

# Returns MediaPipe backend for your system
```

## Future: Making VL-JEPA2 Work on 16GB RAM

If you want to experiment with VL-JEPA2 on your system:

1. **Use ONNX quantization:**
   ```bash
   pip install onnxruntime-extensions
   python -c "from onnxruntime.quantization import quantize_dynamic; quantize_dynamic('vljepa2.onnx', 'vljepa2_quant.onnx')"
   ```

2. **Reduce input resolution:**
   ```python
   # Process at 224x224 instead of 512x512
   input_size=(224, 224)
   ```

3. **Run on CPU with OpenVINO:**
   ```bash
   pip install openvino
   # Convert model to OpenVINO IR format
   ```

4. **Use distilled models:**
   - Consider MobileViT or EfficientNet-based alternatives
   - Trade some accuracy for speed/memory

## Current Recommendation

**For your 16GB RAM + Intel UHD system:**

✅ **Use MediaPipe backend** - It's optimized, fast, and provides good behavior detection for:
- Raised arms (fighting detection)
- Fighting stance
- Running (via trajectory analysis)
- Loitering (via position history)
- Crowd formation (via people count)

❌ **Skip VL-JEPA2 for now** - Wait until you have dedicated GPU or upgrade to 32GB RAM

The MediaPipe + DeepSORT + YOLOv8 combination will give you 10-15 FPS real-time performance with comprehensive suspicious activity detection.

## References

1. LeCun Y. "A Path Towards Autonomous Machine Intelligence." OpenReview, 2022.
2. Meta AI. "V-JEPA: Self-Supervised Learning for Video." ai.meta.com/blog, 2023.
3. Assran M., et al. "Self-Supervised Learning from Images with a Joint Embedding Predictive Architecture." CVPR, 2023.
4. MediaPipe Documentation: https://google.github.io/mediapipe/

---

**Note:** The `behavior_vjepa.py` file in this project contains placeholder code for VL-JEPA2 integration. Full implementation would require:
- PyTorch 2.0+
- Transformers 4.35+
- Model weights from Meta
- GPU with 8GB+ VRAM
