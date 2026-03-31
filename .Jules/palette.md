
## 2024-10-24 - Real-time Telemetry Requires `aria-live` Regions
**Learning:** Real-time UI components like telemetry charts, alert logs, and risk badges that dynamically update their content via WebSockets or polling are not announced by screen readers by default. This causes assistive technology users to miss critical updates like fall detections or risk level changes.
**Action:** Always implement `aria-live` regions (e.g., `aria-live="polite"`) and appropriately configure `aria-relevant` (e.g., `aria-relevant="additions"` for lists where items are added) on dynamic telemetry and alert containers.
