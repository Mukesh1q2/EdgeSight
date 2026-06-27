## 2024-06-01 - FastAPI Async MJPEG Streaming
**Learning:** Synchronous generators in FastAPI `StreamingResponse` run in the AnyIO thread pool, which can block the server under load or exhaust threads. CPU-bound tasks like `cv2.imencode` exacerbate this issue.
**Action:** Always implement MJPEG video frame generators as async generators (`async def`), use `await asyncio.sleep` for throttling, and offload CPU-bound encoding to `asyncio.to_thread`.
