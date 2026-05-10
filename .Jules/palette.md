## 2026-05-10 - Missing Focus States on Custom Interactive Controls
**Learning:** Custom interactive UI controls (like sliders and styled buttons) lack native focus styles and must implement `:focus-visible` outlines to ensure proper keyboard accessibility.
**Action:** Always add explicit `:focus-visible` CSS rules using the design system's primary colors (e.g., `outline: 2px solid var(--primary)`) when styling interactive components to maintain keyboard navigation visibility.
