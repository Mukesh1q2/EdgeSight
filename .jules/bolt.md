## 2024-05-18 - Prevent Event Loop Blocking in FastAPI MJPEG streams
**Learning:** In FastAPI, yielding synchronous generators in StreamingResponse will block the asyncio event loop. Using `time.sleep()` is especially problematic.
**Action:** Implement video frame generators as async generators using `async def`, use `await asyncio.sleep` for throttling, and offload CPU-bound encoding (e.g., `cv2.imencode`) with `asyncio.to_thread` to ensure maximum concurrency.
