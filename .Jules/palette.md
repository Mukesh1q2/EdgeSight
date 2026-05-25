## 2025-05-25 - Custom UI Controls Lack Native Focus
**Learning:** Custom interactive UI controls (like sliders, buttons, and select dropdowns) in this app's "Aegis Vision" design system lack native focus styles, which completely breaks keyboard accessibility indicators when navigating.
**Action:** Always implement explicit `:focus-visible` outlines using existing design tokens (e.g., `var(--primary)`) for all interactive elements to ensure proper keyboard accessibility without affecting mouse users.
