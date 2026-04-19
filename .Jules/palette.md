## 2024-04-19 - Adding Focus-Visible and ARIA-Hidden for Interactive Elements
**Learning:** Interactive elements such as buttons, sliders, and selects must have distinct keyboard focus states (e.g., `:focus-visible`) for accessibility. Additionally, decorative icons inside text-labeled buttons should be hidden from screen readers using `aria-hidden="true"` to prevent redundant or confusing announcements.
**Action:** Always implement `:focus-visible` with a primary outline for interactive components, and verify that decorative icons use `aria-hidden="true"`.
