## 2024-05-24 - Initial Bolt Setup\n**Learning:** Bolt journal must exist for learnings.\n**Action:** Create journal file.
## 2025-02-23 - Avoid datetime.strptime in Hot Loops
**Learning:** Parsing strings into datetime objects inside high-frequency computer vision loops (like `detection_loop` running at ~30 FPS) introduces significant and unnecessary CPU overhead.
**Action:** Use native float timestamps (`time.time()`) and pre-calculated deltas for time-based rate limiting or throttling inside hot loops. Defer string formatting (`strftime`) until after the conditional checks have passed.
