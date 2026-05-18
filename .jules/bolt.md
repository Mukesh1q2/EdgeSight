## 2026-05-18 - String parsing in high-frequency loops
**Learning:** Parsing timestamps from strings (`datetime.strptime`) inside high-frequency loops (like the ~30 FPS detection loop) is a significant and unnecessary CPU bottleneck. The detection loop currently checks cooldowns by parsing the last alert's timestamp string.
**Action:** Use float timestamps (`time.time()`) for rate-limiting and state tracking instead of expensive string parsing and formatting.
