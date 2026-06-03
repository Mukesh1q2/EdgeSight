## 2026-06-03 - Added missing focus-visible styles
**Learning:** The Aegis Vision design system lacked explicit `:focus-visible` styles for its interactive elements (buttons, selects, sliders). Relying on default browser outlines or generic `:focus` is insufficient for keyboard accessibility, as custom styled elements often mask these defaults.
**Action:** Always explicitly define `:focus-visible` states using existing design tokens (e.g., `var(--primary)`) when extending or fixing the Aegis Vision design system to ensure keyboard users have clear visual indicators of their current focus.
