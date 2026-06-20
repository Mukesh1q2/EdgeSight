## 2024-06-20 - [Blocking MJPEG Streams in FastAPI]
**Learning:** Synchronous generators with `time.sleep` and CPU-bound operations (like `cv2.imencode`) in FastAPI MJPEG streams block the threadpool, severely degrading concurrent request latency.
**Action:** Always implement video frame generators as async generators (`async def`), use `await asyncio.sleep` for throttling, and offload CPU-bound encoding using `await asyncio.to_thread`.
