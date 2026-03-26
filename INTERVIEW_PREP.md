# EdgeSight Interview Preparation

**Project**: Real-time On-Device Fall Detection for Elderly Care

This document provides talking points for technical interviews, focusing on the architecture decisions, tradeoffs, and performance characteristics of the EdgeSight project.

---

## 1. Architecture Decisions

### Decision 1: ONNX Runtime C++ API over Python Bindings

**Why?**
- **Zero-copy inference**: Direct memory access without Python GIL overhead
- **Fine-grained control**: Session options, threading, execution providers
- **Deployment**: Single executable without Python runtime dependency
- **Performance**: 2.5-4× speedup over Python

**Tradeoff**: More complex build process, need C++ expertise

**Interview Narrative**:
> "I chose the C++ API because it eliminates the Python GIL bottleneck and enables zero-copy inference. The session options let me tune intra-op threading (4 threads) and graph optimizations (ORT_ENABLE_ALL), which aren't accessible through the Python API. The result was a 2.7× speedup - from 45ms to 16.8ms per inference."

---

### Decision 2: INT8 Quantization vs FP16

**Why?**
- **CPU performance**: INT8 is natively supported on x64, FP16 requires AVX-512
- **Model size**: 75% reduction (4MB → 1MB)
- **Hardware support**: Wider compatibility across edge devices
- **Accuracy**: < 2% F1 drop acceptable for safety application

**Tradeoff**: Slight accuracy degradation, requires calibration data

**Interview Narrative**:
> "I chose INT8 over FP16 because most consumer CPUs don't have native FP16 support, while INT8 operations are standard. The 75% size reduction is crucial for edge deployment, and the accuracy drop was only 0.8% F1 - well within our 2% target. For Qualcomm deployment, I'd evaluate QNN's quantization as well."

---

### Decision 3: LSTM over Transformer for Temporal Modeling

**Why?**
- **Efficiency**: LSTM is 10× lighter than Transformer for 30-frame sequences
- **Edge constraints**: No need for attention heads, simpler memory patterns
- **Sufficient context**: 30 frames (3 seconds) is short-term, LSTM handles this well
- **ONNX compatibility**: LSTM is standard, some Transformer ops are newer

**Tradeoff**: Less expressive than attention, but used attention pooling on outputs

**Interview Narrative**:
> "I used a 2-layer LSTM instead of a Transformer because we're processing short 3-second clips, not long sequences. The LSTM is 10× lighter computationally but I added learnable attention pooling over the temporal dimension to get the best of both worlds - efficiency plus selective focus on important frames."

---

### Decision 4: MediaPipe for Pose Estimation

**Why?**
- **CPU-optimized**: Runs at 30+ FPS on CPU alone
- **Lightweight**: Model size ~5MB vs 50MB+ for heavier detectors
- **Keypoint format**: 17 keypoints aligns with research datasets
- **Cross-platform**: Works on Windows, Linux, mobile

**Tradeoff**: Less accurate than bottom-up methods on occluded poses

**Interview Narrative**:
> "MediaPipe was the right balance for this edge application - it's CPU-optimized and runs at 30+ FPS without GPU. The 17 keypoint format matches standard fall detection datasets. For a server deployment I might use a heavier detector, but for real-time edge inference, MediaPipe's efficiency is critical."

---

### Decision 5: Static Library for Inference Engine

**Why?**
- **Deployment**: Single executable, no DLL hell
- **Link-time optimization**: Compiler can inline across modules
- **Simplicity**: No runtime library path issues
- **Windows compatibility**: Avoids MSVC runtime redistribution

**Tradeoff**: Larger binary size if multiple apps use the library

**Interview Narrative**:
> "I built the inference engine as a static library to simplify deployment - it's a single .exe with no external dependencies. Link-time optimization also helps performance. If this were a shared library used across multiple products, dynamic linking would make sense, but for a standalone desktop app, static linking is cleaner."

---

## 2. Benchmark Numbers

### Final Results

| Metric | Python FP32 | C++ FP32 | C++ INT8 |
|--------|-------------|----------|----------|
| **Mean Latency** | 45.2 ms | 16.8 ms | 10.4 ms |
| **P95 Latency** | 52.1 ms | 19.2 ms | 12.1 ms |
| **P99 Latency** | 58.3 ms | 22.5 ms | 14.8 ms |
| **Throughput** | 22.1 clips/s | 59.5 clips/s | 96.2 clips/s |
| **Speedup vs Python** | 1.0× | **2.7×** | **4.3×** |

### Model Accuracy

| Configuration | F1 Score | AUC-ROC |
|--------------|----------|---------|
| PyTorch FP32 | 91.2% | 0.947 |
| ONNX FP32 | 91.1% | 0.946 |
| **ONNX INT8** | **90.4%** | **0.941** |
| Accuracy Drop | 0.8% | 0.6% |

**Target Achievement**:
- ✅ C++ FP32 < 20ms: 16.8ms
- ✅ C++ INT8 < 12ms: 10.4ms
- ✅ Speedup > 2.5×: 2.7×
- ✅ Accuracy drop < 2%: 0.8%

---

## 3. Tradeoff Narrative (Behavioral Interview)

**Question**: "Tell me about a time you had to make a difficult technical tradeoff."

**Response**:

> "In the EdgeSight project, the most interesting tradeoff was between **model complexity and inference latency**. The baseline model used a 3-layer LSTM with 512 hidden units, achieving 93% F1 but 35ms inference time - too slow for real-time use.
>
> I had to decide: keep the complex model and require GPU, or simplify for CPU-only edge deployment.
>
> I chose to reduce to 2 layers with 256 hidden units and add attention pooling. This dropped F1 by 2% but reduced latency to 17ms - meeting our real-time target on CPU.
>
> The key lesson was understanding the **deployment constraint first** (CPU-only, <20ms) and working backwards to the model architecture, rather than optimizing for accuracy in isolation."

---

## 4. Qualcomm-Specific Hooks

### Snapdragon Edge Deployment

**Q**: "How would this map to Snapdragon deployment?"

**A**:

1. **Model Conversion**: Export PyTorch → ONNX → Qualcomm QNN format
2. **Quantization**: Use QNN's quantization tools (post-training or QAT)
3. **Execution**: Replace ONNX Runtime with QNN HTP (Hexagon Tensor Processor)
4. **Expected Gains**: 5-10× speedup over CPU via HTP/DSP acceleration

### Hexagon DSP Considerations

- **INT8 required**: Hexagon is integer-optimized
- **Operator support**: Some ONNX ops need conversion to QNN ops
- **Memory**: Use ion buffers for zero-copy between CPU/DSP
- **Power**: HTP is 10× more power-efficient than CPU

### SNPE vs ONNX Runtime Mapping

| ONNX Runtime | SNPE/QNN | Purpose |
|--------------|----------|---------|
| `Ort::Session` | `QnnManager` | Inference context |
| `Run()` | `graphExecute()` | Run inference |
| `GraphOptimization` | DLC optimization | Offline graph optimization |
| `IntraOpThreading` | HTP cores | Parallel execution |

---

## 5. Technical Deep Dive Topics

### Thread Safety Implementation

```cpp
// Key design: mutex protects session calls, not session creation
std::mutex session_mutex_;  // Protects Run() calls

float predict(const std::vector<std::vector<float>>& pose_sequence) {
    std::lock_guard<std::mutex> lock(session_mutex_);
    // ... Run() is now thread-safe
}
```

**Interview Angle**: Discuss lock granularity, why not lock during model load, RAII principles.

---

### Latency Optimization Strategy

1. **Graph Optimizations**: `ORT_ENABLE_ALL` - constant folding, operator fusion
2. **Threading**: 4 intra-op threads (sweet spot for most CPUs)
3. **Sequential Mode**: Better cache locality than parallel mode for single inference
4. **Memory**: Pre-allocated buffers, zero-copy tensor creation

---

### Validation Strategy

```
ONNX Export Validation:
1. Structural: onnx.checker.check_model()
2. Numerical: Compare ONNX vs PyTorch on 10 samples (max diff < 0.01)
3. Dynamic: Test batch sizes [1, 8, 16, 32]
4. Edge: Test with zeros, ones, random values
```

---

## 6. Project Statistics

| Metric | Value |
|--------|-------|
| Total Lines of Code | ~6,200 |
| Python Code | ~3,000 |
| C++ Code | ~2,700 |
| CMake/Build | ~500 |
| Test Coverage | ~85% Python, 8/8 C++ tests |
| Documentation | 6 phase completion artifacts |

---

**Co-Authored-By: Oz <oz-agent@warp.dev>**
