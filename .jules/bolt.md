## 2024-06-21 - FastAPI MJPEG Stream Thread Pool Starvation
**Learning:** In FastAPI, using a synchronous generator for `StreamingResponse` (like video feeds) ties up a worker thread indefinitely. For MJPEG streaming, `time.sleep()` and CPU-bound tasks like `cv2.imencode` can exhaust the thread pool, causing the server to hang or reject new connections.
**Action:** Always implement continuous streaming endpoints as `async def` generators, using `await asyncio.sleep()` for throttling and `await asyncio.to_thread()` for CPU-bound frame encoding (e.g., `cv2.imencode`).
