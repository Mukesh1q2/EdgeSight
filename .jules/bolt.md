## 2026-05-02 - [Python String Methods Overhead]
**Learning:** Checking string suffixes against multiple values using a generator inside `any()` (e.g., `any(s.endswith(e) for e in exts)`) introduces significant function call and iteration overhead in hot loops.
**Action:** Always pre-convert collections of extensions to a `tuple` and pass the tuple directly to `str.endswith()` (e.g., `s.endswith(tuple_exts)`), which evaluates at the C level.
