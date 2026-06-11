## 2024-06-11 - Range Slider Focus States
**Learning:** When styling focus states for custom range sliders, apply the `:focus-visible` pseudo-class to the input element itself, not the webkit thumb pseudo-element (e.g., use `.slider:focus-visible::-webkit-slider-thumb` instead of `.slider::-webkit-slider-thumb:focus-visible`).
**Action:** Always ensure focus states are applied to the interactive input elements themselves when modifying custom controls to maintain accessibility.
