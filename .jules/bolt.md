## 2024-05-24 - High-Frequency Loop Time Formatting Anti-Pattern
**Learning:** Using `datetime.strptime()` for timestamp comparison in high-frequency loops (like the ~30 FPS detection loop in `fastapi_server.py`) is surprisingly expensive (over 100x slower than `time.time()`). Also, using time-only formats like `"%H:%M:%S"` with `strptime` defaults the date to 1900-01-01, making `datetime.now() - strptime_result` inaccurate and causing latent bugs.
**Action:** Use float timestamps from `time.time()` for rate limiting and state tracking instead of expensive string parsing and formatting.

## 2024-05-24 - FastAPI MJPEG Stream Thread Exhaustion
**Learning:** Passing a synchronous generator with `time.sleep()` to FastAPI's `StreamingResponse` causes it to run in AnyIO's worker thread pool. While this prevents blocking the main event loop, each connected client permanently consumes a worker thread. If client count exceeds the thread pool limit (default 40), the entire application freezes for new requests.
**Action:** Implement video frame generators as `async def` generators, use `await asyncio.sleep()` for throttling, and offload CPU-bound encoding using `await asyncio.to_thread()` to ensure maximum concurrency and O(1) thread usage.
