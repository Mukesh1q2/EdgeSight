## 2024-05-18 - [FastAPI MJPEG Stream Bottleneck]
**Learning:** In FastAPI, using synchronous generators (with `yield` and `time.sleep()`) for MJPEG video streams block the event loop and prevent concurrency, drastically slowing down other endpoints when clients connect to the stream.
**Action:** Always implement video frame generators for FastAPI as `async def` async generators, use `await asyncio.sleep()` for throttling, and offload CPU-bound work like `cv2.imencode` to threads using `await asyncio.to_thread()`.
