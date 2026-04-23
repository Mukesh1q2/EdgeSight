## 2024-04-23 - Focus States & Decorative Icons
**Learning:** This app's custom UI components (`.btn`, `.slider`) lacked visible focus states for keyboard navigation, and the decorative icons inside buttons were being announced redundantly by screen readers alongside their text labels.
**Action:** Standardized `:focus-visible` styling with a 2px primary outline offset across interactive elements to support keyboard accessibility, and enforced `aria-hidden="true"` on decorative icons inside labeled buttons.
