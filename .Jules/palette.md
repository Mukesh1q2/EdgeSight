## 2026-05-05 - EdgeSight Dashboard Accessibility Patterns
**Learning:** Real-time UI components with low-frequency updates (like alert logs and risk badges) must implement `aria-live` regions to ensure screen readers announce updates. Custom interactive UI controls (like sliders and styled buttons) lack native focus styles and must implement `:focus-visible` outlines for proper keyboard accessibility.
**Action:** Use `aria-live="polite"` on alert containers and add `:focus-visible { outline: 2px solid var(--primary); outline-offset: 2px; }` to custom interactives.
