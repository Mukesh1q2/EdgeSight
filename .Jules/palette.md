## 2026-05-09 - Add focus-visible outlines for keyboard navigation
**Learning:** Custom interactive UI controls (like styled buttons, sliders, and custom selects) lack native focus styles because they override default browser behavior, hiding keyboard focus indicators and reducing accessibility.
**Action:** Always implement explicit `:focus-visible` outlines using the design system's primary color (e.g., `--primary`) when creating or modifying custom UI controls to ensure proper keyboard accessibility.
