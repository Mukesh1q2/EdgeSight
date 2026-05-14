## 2024-05-14 - Python endswith() Performance
**Learning:** Checking string extensions against a collection using a generator expression inside `any()` (e.g., `any(file.endswith(ext) for ext in exts)`) is significantly slower than passing a tuple directly to `str.endswith()` (e.g., `file.endswith(tuple_exts)`), because the latter executes entirely in C rather than executing a Python-level loop.
**Action:** Always use tuples with `endswith()`/`startswith()` for multiple string prefix/suffix checks, especially in loops traversing file systems.
