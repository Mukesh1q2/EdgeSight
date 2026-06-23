## 2024-05-24 - Async Generator Optimization
**Learning:** In `fastapi_server.py`, FastAPI handles MJPEG streaming using generators. If the generator is synchronous (`def` and `time.sleep()`), FastAPI blocks the main event loop, significantly reducing concurrent client throughput.
**Action:** Always use `async def` and `await asyncio.sleep()` for streaming generators in FastAPI to allow the event loop to yield. Additionally, offloading CPU-bound tasks like `cv2.imencode` to threads using `asyncio.to_thread` further maximizes event loop availability.
