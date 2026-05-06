## 2024-05-24 - Asyncio Event Loop Blocking in FastAPI
**Learning:** Background tasks in FastAPI (like `detection_loop`) that perform CPU-bound or blocking I/O operations (e.g., OpenCV video capture, MediaPipe inference, drawing skeletons) synchronously will block the main event loop. This leads to starvation for other async endpoints, such as WebSockets and MJPEG streams.
**Action:** Always offload blocking I/O and CPU-bound functions inside async background loops to worker threads using `asyncio.to_thread` to ensure the application remains responsive.
