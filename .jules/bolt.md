## 2024-06-25 - Use Float Tracking Over String Parsing for High-Frequency Logic
**Learning:** High-frequency event loop checks inside asynchronous applications (like `fastapi_server.py`) can be bottlenecked by string parsing (e.g., `datetime.strptime`). Replacing this with native float arithmetic (`time.time()`) offers massive performance improvements without sacrificing functionality.
**Action:** Always favor basic arithmetic (`float` subtraction) over string datetime manipulations inside core high-frequency loops or detection logic.
