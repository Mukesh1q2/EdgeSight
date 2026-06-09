## 2025-05-15 - Visualizing interactive states for custom inputs
**Learning:** When styling focus states for custom range sliders, the focus is on the input element itself, not pseudo-elements like `::-webkit-slider-thumb`. Applying `:focus-visible` to the input ensures keyboard accessibility.
**Action:** Always apply `:focus-visible` to interactive elements and avoid styling pseudo-elements for the main focus ring to make the application accessible for screen readers and keyboard users.
