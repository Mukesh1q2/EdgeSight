## 2024-07-07 - Synchronous Generators in FastAPI Streams
**Learning:** Using synchronous generators with `time.sleep` and CPU-bound operations (like `cv2.imencode`) in FastAPI `StreamingResponse` endpoints blocks the thread pool, limiting concurrency and potentially blocking the main event loop if pool is exhausted.
**Action:** Always implement video frame generators as `async def` and use `await asyncio.sleep` for throttling. Offload CPU-bound image encoding to a separate thread using `await asyncio.to_thread` to ensure maximum stream concurrency without blocking the server.
