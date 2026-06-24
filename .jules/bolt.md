## 2025-02-23 - [MJPEG Streaming Concurrency]
**Learning:** Using synchronous generators and `time.sleep` in FastAPI StreamingResponse endpoints severely blocks the event loop, degrading concurrency and causing high latency for other API requests during active video streams.
**Action:** Always implement video frame generators as async generators (`async def`), replace `time.sleep` with `await asyncio.sleep`, and offload CPU-bound tasks like `cv2.imencode` using `await asyncio.to_thread` to ensure high concurrency.
