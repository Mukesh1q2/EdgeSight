## 2024-06-25 - Replace expensive datetime parsing in tight loops
**Learning:** Using `datetime.strptime` to parse string timestamps (especially time-only formats like `%H:%M:%S`) inside a tight, high-frequency detection loop causes severe CPU blocking and creates a latent bug when calculations cross midnight.
**Action:** Always use float timestamps (`time.time()`) for high-frequency rate-limiting and time-difference tracking, formatting to strings only for presentation layers or JSON serialization.
