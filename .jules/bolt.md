
## 2024-05-24 - High-Frequency Loop Rate Limiting
**Learning:** Parsing `datetime` strings with `strptime` inside high-frequency loops (e.g., video frame processing at 30 FPS) for rate limiting is a massive performance anti-pattern. Benchmark showed it takes ~0.44s for 10k iterations vs 0.002s for float math.
**Action:** Always use float timestamps (`time.time()`) and simple subtraction for rate-limiting and state tracking in performance-critical loops instead of expensive string parsing and formatting.
