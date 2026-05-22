## 2024-05-22 - Focus Styles for Custom Controls
**Learning:** Custom interactive UI controls (like sliders and styled buttons) lack native focus styles and must implement `:focus-visible` outlines to ensure proper keyboard accessibility.
**Action:** When adding or modifying custom controls, always explicitly add `:focus-visible` styles with a clear outline or box-shadow using the theme's primary color to support keyboard navigation.
