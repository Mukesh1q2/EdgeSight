## 2024-07-06 - Async Video Generators and Event Loop Blocking
**Learning:** In FastAPI, synchronous generators for MJPEG streams (`StreamingResponse`) tie up thread pool workers and can limit concurrency. Additionally, using `time.sleep` and CPU-bound operations like `cv2.imencode` blocks the event loop or thread.
**Action:** Always implement video frame generators as async generators (`async def`), use `await asyncio.sleep` for throttling, and offload CPU-bound image encoding to `asyncio.to_thread` to ensure maximum concurrency and prevent event loop blocking.
