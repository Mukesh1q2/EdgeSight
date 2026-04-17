## 2024-05-18 - [Real-time Component Screen Reader Support]
**Learning:** For low-frequency real-time components (like risk badges or alert logs), using `aria-live="polite"` effectively announces updates to screen readers without overwhelming the user. In contrast, it should be avoided for high-frequency updates (like 10Hz telemetry charts or gauges).
**Action:** Apply `aria-live` thoughtfully based on the frequency of state changes in dynamic dashboards to maintain a usable screen reader experience.
