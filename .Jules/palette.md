## 2025-02-12 - Aria-live Regions on Real-time UI
**Learning:** Adding `aria-live` to high-frequency updating elements (like a 10Hz gauge or chart) overwhelms screen readers. It is only appropriate for low-frequency, significant state changes like alert logs and risk level badges.
**Action:** Always verify the update frequency of dynamic data before applying `aria-live` attributes, limiting them to critical state changes (e.g., `<ul id="alert-list" aria-live="polite">`).
