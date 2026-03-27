## 2024-03-27 - Custom Sliders Need Focus Rings
**Learning:** Applying `outline: none` to range inputs (sliders) in custom dark themes completely hides their keyboard focus state. This creates a severe accessibility trap where keyboard users lose track of where they are on the page.
**Action:** Whenever applying `outline: none` to native inputs, always pair it with a strong `:focus-visible` state using theme primary colors to ensure keyboard a11y.
