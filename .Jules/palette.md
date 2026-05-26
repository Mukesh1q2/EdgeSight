## 2024-05-26 - Missing Keyboard Focus Styles in Aegis Vision
**Learning:** The custom Aegis Vision design system used in this application strips native browser focus styles but fails to implement custom `:focus-visible` states for interactive elements like sliders, selectors, and buttons, breaking keyboard accessibility.
**Action:** Always add explicit `:focus-visible` outlines using existing design tokens (e.g., `var(--primary)`) when creating or modifying interactive controls in this codebase.
