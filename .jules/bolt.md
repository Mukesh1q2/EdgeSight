
## 2024-07-02 - Async MJPEG Generator for Thread Starvation
**Learning:** Synchronous generators used in FastAPI `StreamingResponse` block AnyIO thread pool workers for the duration of the connection. For continuous MJPEG streams, this can cause total thread starvation, causing the server to stall on all other API endpoints.
**Action:** Always implement continuous streaming endpoints as `async def` generators. Use `await asyncio.sleep` for throttling and offload CPU-bound steps (like `cv2.imencode`) using `asyncio.to_thread` to ensure concurrency.
