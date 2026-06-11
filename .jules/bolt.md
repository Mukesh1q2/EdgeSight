## 2024-06-11 - Optimize Alert Rate Limiting

**Learning:** Using `datetime.strptime()` with time-only formats (e.g., "%H:%M:%S") in high-frequency loops (like a video frame processing loop running at 30+ FPS) is extremely CPU-intensive (~200x slower) compared to simple float arithmetic. Additionally, it defaults the date to 1900-01-01, which makes timedelta calculations against `datetime.now()` incorrect and prone to latent bugs, especially around midnight or when long-running processes are involved.

**Action:** For performance-critical loops, especially in real-time edge applications, always track timestamps using float values from `time.time()` for rate limiting, throttling, or cooldown logic. Reserve string formatting (`strftime`) strictly for user-facing display or API payload generation, and only execute it when the rate-limit condition is actually met (e.g., inside the `if` block, rather than unconditionally constructing an alert payload every frame).
