## 2024-05-24 - High-Frequency Loop Optimization
**Learning:** In high-frequency loops (like the `detection_loop` which runs at ~30 FPS), using `datetime.now().strftime()` and `datetime.strptime()` for simple logic like alert debouncing introduces significant string formatting/parsing overhead on every frame.
**Action:** Use float timestamps (`time.time()`) for tracking and rate-limiting instead of strings, saving the string conversion only for when data is actually sent out or stored.
