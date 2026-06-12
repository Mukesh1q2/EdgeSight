## 2023-11-09 - Alert Screen Reader & Webkit Slider Focus
**Learning:** Dynamically updated lists like alerts need `aria-live="assertive"` and `role="log"` to be read by screen readers. Also, when styling custom range sliders, the `:focus-visible` pseudo-class needs to be applied to the `.slider` element itself, targeting the `::-webkit-slider-thumb` pseudo-element.
**Action:** Next time, always check dynamically updated elements for screen reader support and ensure focus states are properly targeted for custom form inputs like range sliders.
