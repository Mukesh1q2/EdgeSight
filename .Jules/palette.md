## 2024-05-24 - Dynamic Telemetry Needs ARIA Live Regions
**Learning:** In real-time dashboards like EdgeSight where alerts and risks update dynamically via WebSockets, these critical changes are completely invisible to screen reader users unless specifically announced.
**Action:** Always wrap high-priority dynamic content (like the Alert Log and Risk Badge) in `aria-live="polite"` (or `"assertive"` if critical) regions so assistive technologies announce updates automatically.
