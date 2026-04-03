## 2026-04-03 - Focus states masked by outline: none
**Learning:** Adding `outline: none` to custom UI components (like the custom range slider) completely removes keyboard focus visibility, harming accessibility for users navigating via keyboard. This is a common pattern in this app's components to suppress default browser styles.
**Action:** Always ensure that if `outline: none` is used, a corresponding `:focus-visible` state is implemented (e.g., using a 2px primary color outline with offset) to restore focus indication for keyboard users while keeping mouse interactions clean.
