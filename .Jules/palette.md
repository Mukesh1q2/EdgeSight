## 2024-11-20 - Ensure Keyboard Focus Indicators for Custom Interactive Elements in Aegis Vision
**Learning:** Custom interactive UI controls (like sliders and styled buttons) in the Aegis Vision design system lack native focus styles (due to styling such as `outline: none`). This breaks keyboard accessibility.
**Action:** When implementing new interactive elements or styling custom inputs (like the threshold slider or gradient buttons), explicitly implement `:focus-visible` outlines using the `--primary` token to ensure proper keyboard accessibility.
