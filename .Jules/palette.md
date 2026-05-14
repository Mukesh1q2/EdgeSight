## 2024-05-14 - Interactive Controls Lack Native Focus Styling
**Learning:** Custom UI elements like styled range sliders and custom buttons in this app often have their native outlines removed (e.g., `outline: none` or through CSS resets). This creates a critical accessibility issue where keyboard users cannot perceive focus states.
**Action:** Always implement explicit `:focus-visible` styles using the existing design tokens (like `var(--primary)`) whenever customizing interactive components that would otherwise lose default browser focus outlines.
