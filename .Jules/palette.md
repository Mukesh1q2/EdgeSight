## 2024-04-27 - Keyboard Navigation Focus Indicators
**Learning:** Custom UI components like `.btn`, `.selector select`, and `.slider` in this design system lacked explicit `:focus-visible` states, making keyboard navigation difficult to track visually for accessibility users.
**Action:** Always add standard explicit `:focus-visible` states (e.g., `outline: 2px solid var(--primary); outline-offset: 2px;`) to all interactive elements to ensure compliance with keyboard accessibility standards.
