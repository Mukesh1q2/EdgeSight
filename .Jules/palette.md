## 2024-05-12 - Explicit Focus Styles for Aegis Vision Custom Controls
**Learning:** Custom interactive controls (buttons, sliders, selects) using the Aegis Vision CSS design system do not inherit native browser focus outlines. This breaks keyboard accessibility because users cannot see which element is currently focused when tabbing through the UI.
**Action:** Always add explicit `:focus-visible` states using existing design tokens (e.g., `outline: 2px solid var(--primary); outline-offset: 2px;`) whenever implementing or modifying custom interactive controls in this project.
