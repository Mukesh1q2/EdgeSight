## 2024-04-09 - Defer expensive string formatting in high-frequency loops
**Learning:** Creating string-formatted datetimes or parsing them inside a 30 FPS computer vision hot loop (`detection_loop`) wastes significant CPU cycles when evaluating rate limits (which drop most frames).
**Action:** Optimize performance in hot loops by deferring computationally expensive operations like string formatting (e.g., datetime.strftime) until after conditional checks or rate limits have passed. Use native timestamps (like time.time()) for threshold comparisons.
