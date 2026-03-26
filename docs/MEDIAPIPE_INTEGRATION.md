# MediaPipe Integration Guide

This document explains how to integrate real pose detection using MediaPipe into EdgeSight.

## Current Implementation

The current codebase uses **simulated pose detection** in `app/processing_thread.cpp`:

```cpp
bool ProcessingThread::extractPoseKeypoints(const cv::Mat& frame, std::vector<float>& keypoints) {
    // SIMULATED - returns hardcoded keypoints
    // Replace with real MediaPipe integration
}
```

This is sufficient for testing the inference pipeline but should be replaced with real pose detection for production use.

## Integration Options

### Option 1: MediaPipe C++ API (Recommended for Production)

**Difficulty**: Hard  
**Performance**: Excellent  
**Latency**: Lowest

#### Steps:

1. **Install MediaPipe C++**
   ```bash
   # Clone MediaPipe
   git clone https://github.com/google/mediapipe.git
   cd mediapipe
   
   # Build C++ libraries (requires Bazel)
   bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 \
       mediapipe/examples/desktop/pose_tracking:pose_tracking_cpu
   ```

2. **Link to EdgeSight**
   
   Add to `inference/engine/CMakeLists.txt`:
   ```cmake
   find_package(mediapipe REQUIRED)
   target_link_libraries(fall_detector_lib mediapipe::pose_detection)
   ```

3. **Update `processing_thread.cpp`**
   ```cpp
   #include "mediapipe/framework/calculator_graph.h"
   #include "mediapipe/framework/formats/landmark.pb.h"
   
   bool ProcessingThread::extractPoseKeypoints(const cv::Mat& frame, std::vector<float>& keypoints) {
       // Run MediaPipe graph
       mediapipe::CalculatorGraph graph;
       // ... initialize graph ...
       
       // Convert output landmarks to keypoints
       auto landmarks = output_packet.Get<std::vector<mediapipe::NormalizedLandmark>>();
       for (const auto& lm : landmarks) {
           keypoints.push_back(lm.x());
           keypoints.push_back(lm.y());
           keypoints.push_back(lm.visibility());
       }
       return keypoints.size() == 45;  // 15 keypoints * 3 values
   }
   ```

### Option 2: Python Subprocess (Easiest)

**Difficulty**: Easy  
**Performance**: Good  
**Latency**: Medium (process spawn overhead)

#### Steps:

1. **Create Python pose extractor** (`scripts/mediapipe_pose.py`):
   ```python
   import sys
   import cv2
   import mediapipe as mp
   import json
   import numpy as np
   
   mp_pose = mp.solutions.pose
   pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
   
   def extract_keypoints(image_path):
       image = cv2.imread(image_path)
       results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
       
       keypoints = []
       if results.pose_landmarks:
           for lm in results.pose_landmarks.landmark:
               keypoints.extend([lm.x, lm.y, lm.visibility])
       return keypoints
   
   if __name__ == "__main__":
       # Read image from stdin or file
       image_path = sys.argv[1]
       keypoints = extract_keypoints(image_path)
       print(json.dumps(keypoints))
   ```

2. **Call from C++**:
   ```cpp
   bool ProcessingThread::extractPoseKeypoints(const cv::Mat& frame, std::vector<float>& keypoints) {
       // Save frame to temp file
       std::string temp_path = "/tmp/frame.jpg";
       cv::imwrite(temp_path, frame);
       
       // Run Python script
       std::string cmd = "python scripts/mediapipe_pose.py " + temp_path;
       std::array<char, 4096> buffer;
       std::string result;
       
       FILE* pipe = popen(cmd.c_str(), "r");
       while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
           result += buffer.data();
       }
       pclose(pipe);
       
       // Parse JSON output
       auto json_array = nlohmann::json::parse(result);
       for (float val : json_array) {
           keypoints.push_back(val);
       }
       return keypoints.size() == 45;
   }
   ```

### Option 3: ONNX Runtime (Lightweight)

**Difficulty**: Medium  
**Performance**: Very Good  
**Latency**: Low

Use MediaPipe's pretrained ONNX models directly:

1. **Download pretrained models**:
   ```bash
   wget https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
   ```

2. **Extract ONNX model** from task file (it's a zip)

3. **Run inference** using existing ONNX Runtime infrastructure

## Keypoint Mapping

MediaPipe outputs 33 keypoints, but EdgeSight expects 15. Map as follows:

| MediaPipe Index | EdgeSight Index | Body Part |
|-----------------|-----------------|-----------|
| 0 | 0 | Nose |
| 2 | 1 | Left eye |
| 5 | 2 | Right eye |
| 11 | 3 | Left shoulder |
| 12 | 4 | Right shoulder |
| 13 | 5 | Left elbow |
| 14 | 6 | Right elbow |
| 15 | 7 | Left wrist |
| 16 | 8 | Right wrist |
| 23 | 9 | Left hip |
| 24 | 10 | Right hip |
| 25 | 11 | Left knee |
| 26 | 12 | Right knee |
| 27 | 13 | Left ankle |
| 28 | 14 | Right ankle |

**Implementation**:
```cpp
const int MEDIAPIPE_TO_EDGESIGHT[] = {
    0,   // nose -> nose
    2,   // left eye -> left eye
    5,   // right eye -> right eye
    11,  // left shoulder -> left shoulder
    12,  // right shoulder -> right shoulder
    13,  // left elbow -> left elbow
    14,  // right elbow -> right elbow
    15,  // left wrist -> left wrist
    16,  // right wrist -> right wrist
    23,  // left hip -> left hip
    24,  // right hip -> right hip
    25,  // left knee -> left knee
    26,  // right knee -> right knee
    27,  // left ankle -> left ankle
    28   // right ankle -> right ankle
};
```

## Testing Integration

After implementing real pose detection:

1. **Test with synthetic video**:
   ```bash
   python scripts/generate_synthetic_data.py --visualize
   ```

2. **Compare outputs**:
   - Simulated: Should return predictable keypoints
   - Real: Should detect actual human poses

3. **Performance benchmark**:
   ```bash
   python inference/benchmark/benchmark.py
   ```

## Expected Performance

With MediaPipe integration, expect:

| Configuration | Pose Detection | Inference | Total Latency |
|--------------|----------------|-----------|---------------|
| Simulated | ~1ms | ~5ms | ~6ms |
| MediaPipe Light | ~10ms | ~5ms | ~15ms |
| MediaPipe Heavy | ~30ms | ~5ms | ~35ms |

**Target**: Keep total latency under 50ms for real-time detection.

## Troubleshooting

### High CPU Usage

- Use MediaPipe `POSE_LANDMARKS_LITE` model
- Reduce input resolution to 320x240
- Skip frames (process every 2nd or 3rd frame)

### Poor Detection Accuracy

- Use `POSE_LANDMARKS_HEAVY` model
- Ensure good lighting conditions
- Check camera focus

### Build Errors

- MediaPipe C++ requires Bazel build system
- Consider Option 2 (Python subprocess) for faster development

## Recommended Approach

For **MVP/Development**: Use Option 2 (Python subprocess)  
For **Production**: Use Option 1 (Native C++ MediaPipe)

---

**Co-Authored-By: Oz <oz-agent@warp.dev>**
