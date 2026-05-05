## 2024-05-24 - Avoiding String Parsing in Computer Vision Hot Loops
**Learning:** Performing datetime string parsing (`strptime`) and formatting (`strftime`) inside a high-frequency (30 FPS) computer vision loop causes unnecessary CPU overhead and can lead to frame drops or delayed alerts.
**Action:** Always use float timestamps (`time.time()`) for rate-limiting and state tracking, and defer expensive string operations until after conditional checks or rate limits have passed.
