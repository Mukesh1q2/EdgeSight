## 2026-07-04 - [FastAPI MJPEG Streaming Concurrency]
**Learning:** In FastAPI MJPEG streams, CPU-bound encoding (e.g., `cv2.imencode`) combined with synchronous generators blocks the event loop, preventing maximum concurrency and smooth real-time performance.
**Action:** Always implement video frame generators as async generators (`async def`), use `await asyncio.sleep` for throttling, and offload CPU-bound encoding with `asyncio.to_thread`.
