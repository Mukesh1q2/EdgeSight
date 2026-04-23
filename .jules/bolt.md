## 2024-04-23 - Avoid computationally expensive string operations in hot loops
**Learning:** In a high-frequency (30 FPS) computer vision hot loop like `detection_loop`, using computationally expensive string operations such as `datetime.strptime()` for evaluating alert cooldowns creates unnecessary performance overhead.
**Action:** Replace `datetime` operations with native float comparisons using `time.time()`, and defer any string formatting (like `datetime.strftime()`) until after the conditional logic/cooldown check has passed.
