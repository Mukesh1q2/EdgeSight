## 2024-05-18 - [Prevent Event Loop Blocking in MJPEG Streams]
**Learning:** Synchronous generator functions used in FastAPI/Starlette StreamingResponse can block the event loop or exhaust threadpools, especially when doing CPU-bound work like `cv2.imencode` and using blocking `time.sleep()`.
**Action:** Always implement video frame streams as `async def` generators, use `await asyncio.sleep` for throttling, and offload CPU-bound image encoding to worker threads using `await asyncio.to_thread`.
