## 2024-06-26 - Async Generator for MJPEG Streaming
**Learning:** Using synchronous generators with `time.sleep()` for MJPEG streams in FastAPI consumes worker threads, risking threadpool exhaustion under concurrent load.
**Action:** Always use `async def` and `await asyncio.sleep()` for video streaming endpoints, and offload CPU-heavy tasks like `cv2.imencode` using `asyncio.to_thread()`.
