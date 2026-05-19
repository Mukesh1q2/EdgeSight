## 2024-05-19 - Missing Focus Visible on Custom Controls
**Learning:** The Aegis Vision design system aggressively strips native focus outlines (`outline: none`) on custom controls like sliders and buttons, making them completely inaccessible to keyboard-only users who rely on visual indicators.
**Action:** Always verify that interactive elements with `outline: none` or custom styles have an explicit `:focus-visible` fallback (e.g., `outline: 2px solid var(--primary)`) implemented.
