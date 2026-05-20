## 2026-05-20 - Add explicit :focus-visible outlines to custom UI elements
**Learning:** Custom interactive UI controls (like sliders and styled buttons) in the Aegis Vision design system lack native focus styles, which hides keyboard navigation state.
**Action:** Always implement explicit `:focus-visible` outlines using the `--primary` design token when building or updating custom UI controls to ensure proper keyboard accessibility.
