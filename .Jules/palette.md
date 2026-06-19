## 2024-06-19 - Added ARIA live regions for real-time alerts
**Learning:** Real-time dashboards updating via websockets or loops need `aria-live` on containers (like `role="log"`) so that dynamic content like fall detection alerts are announced by screen readers without requiring a page reload.
**Action:** Always verify dynamic content updates have `aria-live` regions when implementing or reviewing dashboard UI components.
