# Bolt's Journal
## 2024-05-18 - Replacing `strptime` with `time.time()` for Cooldown Checks
**Learning:** `datetime.strptime()` is notoriously slow, and doing it per-frame in a high-frequency real-time detection loop inside FastAPI blocks CPU. Additionally, using `datetime.strptime` with time-only formats assumes the date 1900-01-01, which makes delta calculations against `datetime.now()` incorrect without further adjustments.
**Action:** Use `time.time()` for fast float arithmetic when managing time states like cooldowns and rate limits in performance-critical code paths.
