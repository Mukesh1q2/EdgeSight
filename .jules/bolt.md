## 2024-05-24 - Async MJPEG streaming
**Learning:** In FastAPI, synchronous generators passed to `StreamingResponse` (like `def generate_video_frames()`) containing `time.sleep()` and CPU-bound operations (like `cv2.imencode`) block the main asyncio event loop, causing all other API endpoints to stall.
**Action:** Always implement video streaming generators as `async def` (async generators), use `await asyncio.sleep()` for throttling, and offload synchronous CPU-bound operations like `cv2.imencode` with `asyncio.to_thread`.
