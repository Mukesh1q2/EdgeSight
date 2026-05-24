## 2024-05-24 - Focus Visible Outlines Missing for Interactive Elements
**Learning:** Custom interactive UI controls in this project's design system (like the styled `.btn` and `.slider`) lack native focus styles, making them inaccessible for keyboard navigation.
**Action:** When adding new interactive components or updating existing ones in the CSS, explicitly add `:focus-visible` outline styles using existing theme variables (e.g., `outline: 2px solid var(--primary)`) to ensure keyboard accessibility while preventing mouse-click focus rings.
