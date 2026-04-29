## 2024-05-24 - Initial File\n**Learning:** Started journal\n**Action:** Create when missing

## 2024-05-24 - Hot Loop Time Parsing Bottleneck
**Learning:** In the `detection_loop` which runs at 30 FPS, checking the alert cooldown used `datetime.strptime()` to parse a string timestamp back into a datetime object for comparison. This is a massive overhead in a hot loop that gets evaluated every 33ms during a fall detection event. Additionally, the string formatting `datetime.now().strftime()` was happening before the cooldown check, so it was executed on every frame even when no alert was generated.
**Action:** Always prefer native float timestamps (`time.time()`) for hot loop delta calculations. Defer expensive string formatting operations until *after* rate-limiting or conditional checks have passed so they only run when absolutely necessary.
