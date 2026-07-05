## 2024-05-24 - [Avoid blocking event loops]
**Learning:** cv2.imencode is CPU bound and blocks asyncio event loop when streaming MJPEG video
**Action:** Use asyncio.to_thread to run cv2.imencode in a separate thread.
