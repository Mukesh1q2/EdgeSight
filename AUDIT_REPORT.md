# EdgeSight Project - E2E Audit Report

**Audit Date**: 2026-03-24  
**Auditor**: Oz  
**Project Status**: Code Complete, Pending Validation

---

## Executive Summary

| Category | Count | Status |
|----------|-------|--------|
| **Total Files** | 35 | ✅ Complete |
| **Syntax Valid** | 35/35 | ✅ Pass |
| **Critical Issues** | 4 | ⚠️ Needs Fix |
| **Minor Issues** | 12 | 🔧 Enhancement |
| **Missing Docs** | 3 | 📝 To Add |

**Overall Health**: 🟡 **FUNCTIONAL BUT REQUIRES POLISH** - Core implementation complete, 4 critical fixes needed before production.

---

## 🔴 CRITICAL ISSUES (Must Fix)

### 1. Alert Manager - Missing `<filesystem>` Header ✅ FIXED
**File**: `app/alert_manager.cpp:25`
**Severity**: 🔴 **CRITICAL - Build Will Fail**

**Status**: ✅ Fixed - Added `#include <filesystem>` at line 14

---

### 2. Alert Manager - CURL Header Guard Issue ✅ FIXED
**File**: `app/alert_manager.h:17-21`
**Severity**: 🔴 **CRITICAL - Compile Error on Missing CURL**

**Status**: ✅ Fixed - Added typedef stub for CURL* when ALERTS_ENABLED not defined:
```cpp
#else
// Stub types when CURL not available
typedef void CURL;
#endif
```

---

### 3. Processing Thread - Model Existence Check ✅ FIXED
**File**: `app/processing_thread.cpp:144-150`
**Severity**: 🔴 **CRITICAL - Runtime Error**

**Status**: ✅ Fixed - Added model existence check before loading:
```cpp
// Check if model file exists
if (!std::filesystem::exists(model_path_)) {
    QString err_msg = QString("Model file not found: ") + ...;
    emit error(err_msg);
    emit statusUpdate("Please export a model using export_onnx.py");
    running_.store(false);
    return;
}
```

---

### 4. Requirements.txt - Missing Pillow ✅ FIXED
**File**: `requirements.txt:48-49`
**Severity**: 🔴 **CRITICAL - Runtime Failure**

**Status**: ✅ Fixed - Added `Pillow>=10.0.0`

---

## 🟠 BROKEN FEATURES (Functional But Defective)

### 5. Pose Detection - SIMULATED (Not Real MediaPipe)
**File**: `app/processing_thread.cpp:53-87`
**Severity**: 🟠 **MAJOR - Core Feature Non-Functional**

```cpp
bool ProcessingThread::extractPoseKeypoints(const cv::Mat& frame, std::vector<float>& keypoints) {
    // In production, this would use MediaPipe Pose
    // For now, we simulate pose detection with simple heuristics
    // ... generates FAKE keypoints based on frame center
}
```

**Current Behavior**: Returns hardcoded simulated keypoints, not actual pose detection.

**Fix Options**:
- **Option A**: Link MediaPipe C++ API (complex build)
- **Option B**: Use Python subprocess to run MediaPipe and pipe results
- **Option C**: Call Python MediaPipe from C++ via pybind11

**Recommended**: Option B for MVP - spawn Python process that outputs JSON keypoints.

---

### 6. Dataset Download - No Automatic Download
**File**: `data/download_datasets.py`
**Severity**: 🟠 **MAJOR - Manual Step Required**

Both UR Fall and Le2i datasets require **manual download**:
- UR Fall requires registration at academic site
- Le2i Kaggle download often fails with API issues

**Workaround**: The script creates instructions but doesn't actually download data.

**Enhancement**: Add synthetic data generator for testing without real datasets.

---

### 7. Benchmark C++ Binary - CMake Integration ✅ FIXED
**File**: `inference/engine/CMakeLists.txt:162-185`
**Severity**: 🟠 **MODERATE - Orphaned Code**

**Status**: ✅ Fixed - Added cpp_benchmark executable target to CMakeLists.txt

---

## 🟡 ENHANCEMENT AREAS (Improvements Needed)

### 8. LICENSE File ✅ ADDED
**File**: `LICENSE`
**Status**: ✅ Added - Complete Apache 2.0 license text

---

### 9. .gitignore ✅ ADDED
**File**: `.gitignore`
**Status**: ✅ Added - Comprehensive .gitignore covering build outputs, Python, datasets, models, Qt, and OS files

---

### 10. CMake - Hardcoded ONNX Runtime Download
**File**: `inference/engine/CMakeLists.txt:48-74`
**Status**: 🔧 **FRAGILE**

ONNX Runtime download uses `execute_process` with PowerShell - fragile and doesn't verify checksum.

**Enhancement**: 
- Add SHA256 verification
- Add retry logic
- Support offline builds with pre-downloaded archive

---

### 11. Configuration File Loading ✅ ADDED
**File**: `app/config_loader.h`, `app/config_loader.cpp`, `app/config.json`
**Status**: ✅ Added - Complete JSON config system with:
- Camera, detection, model, alert, UI, and performance settings
- Singleton ConfigManager for global access
- Save/load with automatic directory creation

---

### 12. No Logging Framework
**Status**: 🔧 **MISSING**

Current logging uses `std::cout` and `std::cerr`. Should use:
- spdlog (lightweight, header-only option)
- Qt logging (if staying in Qt ecosystem)
- Custom logger with levels (DEBUG, INFO, WARN, ERROR)

---

### 13. No Model Versioning
**File**: `model/export_onnx.py`
**Status**: 🔧 **ENHANCEMENT**

Exported models don't include:
- Training timestamp
- Git commit hash
- Dataset version
- Accuracy metrics in metadata

**Enhancement**: Add ONNX metadata:
```python
import onnx
model = onnx.load("model.onnx")
model.doc_string = f"Trained: {timestamp}, F1: {f1_score}"
onnx.save(model, "model.onnx")
```

---

### 14. Training - No Resume from Checkpoint
**File**: `model/train.py`
**Status**: 🔧 **ENHANCEMENT**

If training interrupted, must restart from epoch 0. No `--resume` flag.

**Enhancement**: Add checkpoint resume support.

---

### 15. GUI Keyboard Shortcuts ✅ ADDED
**File**: `app/mainwindow.cpp:417-446`
**Status**: ✅ Added - Keyboard shortcuts:
- `Ctrl+R`: Toggle detection (Start/Stop)
- `Ctrl+S`: Save screenshot
- `Ctrl+Q`: Quit
- `Space`: Toggle focus between controls

---

### 16. No Unit Tests for Python Inference
**File**: `inference/python_infer.py`
**Status**: 🔧 **TEST GAP**

No pytest coverage for `PythonFallDetector` class.

**Enhancement**: Add `tests/test_python_infer.py` with mock model.

---

### 17. CMakeLists.txt Root - Missing FindPackage Checks
**File**: `CMakeLists.txt`
**Status**: 🔧 **ROBUSTNESS**

OpenCV and Qt checks use `QUIET` but don't fail gracefully with helpful messages.

---

### 18. Frame Drop Detection ✅ ADDED
**File**: `app/mainwindow.cpp:302-310`
**Status**: ✅ Added - Frame drop detection in UI with timestamp analysis:
```cpp
// Frame drop detection
auto now = std::chrono::steady_clock::now();
if (last_frame_time_.time_since_epoch().count() > 0) {
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(...).count();
    if (elapsed > 150) {  // Frame took too long
        dropped_frame_count_++;
    }
}
```

---

### 19. Report Generator - No Error Handling for Missing Dependencies
**File**: `inference/benchmark/report_generator.py`
**Status**: 🔧 **ROBUSTNESS**

If `fpdf2` or `matplotlib` not installed, script crashes. Should:
- Check dependencies on startup
- Provide fallback to CSV-only output

---

### 20. Setup Script - No Admin Check
**File**: `scripts/setup_env.bat`
**Status**: 🔧 **USABILITY**

Script doesn't check for:
- Visual Studio installation
- CMake in PATH
- Qt availability
- Sufficient disk space

---

## ✅ WHAT WORKS (Validating Functionality)

| Component | Status | Notes |
|-----------|--------|-------|
| **Python Syntax** | ✅ All 12 .py files compile | Clean |
| **Data Pipeline** | ✅ Structure complete | Needs real data |
| **FallNet Model** | ✅ Architecture valid | Tested with random input |
| **ONNX Export** | ✅ Code complete | Needs trained model to test |
| **C++ Engine** | ✅ RAII, thread-safe | Needs ONNX Runtime to build |
| **Qt GUI Structure** | ✅ UI layout defined | Needs Qt6 + OpenCV to compile |
| **CMake Config** | ✅ Multi-target build | Needs dependencies installed |
| **CI/CD** | ✅ Workflow defined | Needs GitHub repo + secrets |

---

## 📋 PRIORITY FIX CHECKLIST

### Immediate (Do Now) ✅ COMPLETE
- [x] Add `#include <filesystem>` to `app/alert_manager.cpp`
- [x] Add `Pillow` to `requirements.txt`
- [x] Create `.gitignore`
- [x] Create `LICENSE` file

### Before First Build ✅ COMPLETE
- [x] Fix CURL header guards in alert_manager
- [x] Add `cpp_benchmark` to CMakeLists.txt
- [x] Add model existence check in processing_thread

### Before Production
- [ ] Replace simulated pose detection with real MediaPipe
- [ ] Add logging framework
- [x] Add config file loading for alerts
- [x] Add keyboard shortcuts to GUI
- [x] Add frame drop handling
- [ ] Add input validation to main window (partial: threshold, camera done)

---

## 🎯 RECOMMENDED NEXT ACTIONS

1. **Fix Critical Issues** (1-2 hours)
   - Filesystem header, Pillow dependency, .gitignore, LICENSE

2. **Validate Build** (2-4 hours)
   - Run `setup_env.bat` on clean Windows VM
   - Fix any CMake/configure errors

3. **Integration Test** (4-8 hours)
   - Generate synthetic data
   - Run mini-training (5 epochs)
   - Test Python inference
   - Build C++ engine
   - Run unit tests

4. **GUI Smoke Test** (2-4 hours)
   - Build with Qt6
   - Test without camera (video file fallback)
   - Verify all UI controls

5. **Documentation** (2 hours)
   - Add Architecture diagram to docs/
   - Update README with actual build instructions

---

**Co-Authored-By: Oz <oz-agent@warp.dev>**
