---
name: Precision Vision
colors:
  surface: '#fbf8ff'
  surface-dim: '#dad9e3'
  surface-bright: '#fbf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f4f2fd'
  surface-container: '#eeedf7'
  surface-container-high: '#e8e7f1'
  surface-container-highest: '#e3e1ec'
  on-surface: '#1a1b22'
  on-surface-variant: '#464554'
  inverse-surface: '#2f3038'
  inverse-on-surface: '#f1effa'
  outline: '#777586'
  outline-variant: '#c7c4d7'
  surface-tint: '#5148d7'
  primary: '#2a14b4'
  on-primary: '#ffffff'
  primary-container: '#4338ca'
  on-primary-container: '#c1beff'
  inverse-primary: '#c3c0ff'
  secondary: '#5654a8'
  on-secondary: '#ffffff'
  secondary-container: '#a7a5ff'
  on-secondary-container: '#393689'
  tertiary: '#692400'
  on-tertiary: '#ffffff'
  tertiary-container: '#8f3400'
  on-tertiary-container: '#ffb393'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e3dfff'
  primary-fixed-dim: '#c3c0ff'
  on-primary-fixed: '#100069'
  on-primary-fixed-variant: '#372abf'
  secondary-fixed: '#e2dfff'
  secondary-fixed-dim: '#c3c0ff'
  on-secondary-fixed: '#100563'
  on-secondary-fixed-variant: '#3e3c8f'
  tertiary-fixed: '#ffdbcd'
  tertiary-fixed-dim: '#ffb597'
  on-tertiary-fixed: '#360f00'
  on-tertiary-fixed-variant: '#7d2d00'
  background: '#fbf8ff'
  on-background: '#1a1b22'
  surface-variant: '#e3e1ec'
typography:
  display-lg:
    fontFamily: Manrope
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.025em
  display-lg-mobile:
    fontFamily: Manrope
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Manrope
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Manrope
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 26px
    letterSpacing: -0.015em
  title-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '600'
    lineHeight: 22px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
    letterSpacing: 0em
  label-md:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  code-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: -0.01em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  space-1: 0.25rem
  space-2: 0.5rem
  space-3: 0.75rem
  space-4: 1rem
  space-6: 1.5rem
  space-8: 2rem
  space-12: 3rem
  space-16: 4rem
  gutter-desktop: 1.5rem
  margin-desktop: 2rem
  gutter-mobile: 1rem
  margin-mobile: 1rem
---

## Brand & Style

This design system serves a technical, high-rigor machine learning interface designed for engineers, researchers, and data practitioners interacting with computer vision inference models. The aesthetic is inspired by the meticulous discipline of Linear, Vercel, and Raycast: utilitarian elegance, surgical clarity, and zero decorative fluff. 

Every pixel communicates functional state or data provenance. The UI avoids over-stylized tropes—no saturated neon accents, no heavy glowing outlines, and no excessive frosted glass layers. The emotional tone is calm, trustworthy, and authoritative, elevating the visual output of image classification without competing with it. Spatial precision, micro-typography, crisp single-pixel rules, and tactile yet muted transitions define the interaction rhythm.

## Colors

The palette is engineered for prolonged analytical sessions with minimal eye strain, leaning on warm, neutral substrates that recede behind visual inference artifacts.

- **Canvas & Surfaces:** The root canvas uses warm stone `#FAFAF9`, establishing an editorial, tactile foundation. Active work surfaces, inspection panels, and floating cards render in pure `#FFFFFF` to provide distinct structural separation without aggressive contrast jumps.
- **Typography Scale:** Primary body and heading copy is set to `#18181B` (Zinc 900) for maximum legibility. Supporting metadata, secondary labels, and model parameters use `#71717A` (Zinc 500). Tertiary captions, disabled values, and placeholders use `#A1A1AA` (Zinc 400).
- **Structural Borders:** Surface separation relies strictly on a calibrated 1px hairline rule `#E4E4E7` (Zinc 200). Subtle hover states step to `#D4D4D8` (Zinc 300).
- **Functional Accents:** Primary intent is governed by deep architectural indigo (`#4338CA`), descending to `#3730A3` on hover and `#312E81` for pressed or high-emphasis states. Accent color is applied sparingly: primary execution buttons, selected navigation pills, and focused precision controls only.
- **Feedback & Metrics:** Precision validation states use desaturated, authoritative tones: `#15803D` (Emerald 700) for confidence thresholds and verified ground truth, and `#B91C1C` (Red 700) for misclassifications or inference failures.

## Typography

The type scale combines **Manrope** for structured headings and **Inter** for dense, readable UI content.

Headings leverage tight tracking (`-0.02em` to `-0.025em`) to evoke the precision of contemporary developer workbenches. Body text and numerical data rely on Inter’s optimized optical legibility, set with tabular lining figures (`font-feature-settings: 'tnum' on, 'cv05' on`) to ensure inference probabilities, class ranks, and millisecond metrics align predictably across rows.

Dense data readouts such as CIFAR-100 class hierarchy taxonomies (`coarse_label / fine_label`) utilize `label-sm` and `code-sm` to maintain balance in constrained panels.

## Layout & Spacing

Layouts adhere to a 4px baseline sub-grid unified under an 8-point structural rhythm (`4px`, `8px`, `12px`, `16px`, `24px`, `32px`, `48px`, `64px`). 

- **Layout Structure:** A fluid shell bounded at `1440px` maximum width. Standard views implement an asymmetrical split: a 480px fixed-utility left rail (for sample ingest, pre-processing toggles, model weights, and inference controls) alongside a fluid primary canvas for image inspection, activation maps, and probability histograms.
- **Breakpoints & Reflow:**
  - **Desktop (>= 1024px):** Dual-pane persistent inspector layout with 24px gutters and 32px canvas inset padding.
  - **Tablet (768px – 1023px):** Collapsible lateral control drawer; inspection canvas collapses to a single stacked column with 16px gutters.
  - **Mobile (< 768px):** Linear stacked feed. Controls convert to contextual bottom sheets, ensuring the test image and primary prediction badge remain visible without viewport panning.

## Elevation & Depth

Visual hierarchy is maintained through crisp structural containment, not exaggerated vertical floating. The elevation stack relies almost entirely on single-pixel zinc borders paired with micro-ambient drop shadows.

- **Base Surfaces (Canvas):** Grounded at depth 0 on `#FAFAF9` with zero shadow.
- **Level 1 (Cards, Modules, Sidebars):** Pure `#FFFFFF` background bound by a 1px solid `#E4E4E7` border. Depth is signaled using an imperceptible resting shadow: `box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05)`.
- **Level 2 (Dropdowns, Command Menus, Flyouts):** Pure `#FFFFFF` floating layer with a 1px solid `#E4E4E7` border, backed by a structured dual-layer drop: `box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04)`.
- **Level 3 (Modal Overlays):** `#FFFFFF` anchored over a high-density, low-luminance neutral backdrop (`rgba(24, 24, 27, 0.35)` with a 4px blur). Shadow: `box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 8px 10px -6px rgba(0, 0, 0, 0.03)`.

## Shapes

The design system maintains a calibrated progression of border radii mapped strictly to physical element footprint:

- **Interactive Micro-controls & Inputs (8px / `rounded-lg`):** Form controls, text inputs, segment toggles, and data filters use an exact 8px corner radius to stay taut and architectural.
- **Action Buttons (9px–10px):** Primary and secondary trigger buttons receive a slightly softer 10px radius to provide a comfortable tactile target without feeling rounded or pill-shaped.
- **Cards & Data Modules (12px–14px / `rounded-xl`):** Prediction score cards, matrix containers, and metadata tables use a 12px or 14px boundary.
- **Upload Dropzones & Canvas Previews (16px / `rounded-2xl`):** Large visual components such as the image dropzone and heat-map inspection canvases employ a 16px corner radius, balancing larger physical bounding boxes with smooth exterior curves.
- **Pills / Status Badges (Full Radius):** Reserved solely for categorical class indicators, confidence scores, and latency readouts (`9999px`).

## Components

### Buttons
- **Primary:** Background `#4338CA`, text `#FFFFFF`, radius 10px, padding 8px 16px. Resting shadow: `0 1px 2px 0 rgba(0, 0, 0, 0.1)`. Inset micro-highlight on top border: `inset 0 1px 0 0 rgba(255, 255, 255, 0.15)`. Hover: `#3730A3`. Active: `#312E81`.
- **Secondary:** Background `#FFFFFF`, text `#18181B`, border 1px solid `#E4E4E7`, radius 10px, padding 8px 16px. Resting shadow: `0 1px 2px 0 rgba(0, 0, 0, 0.05)`. Hover: `#FAFAF9` with border `#D4D4D8`.
- **Ghost:** Background transparent, text `#71717A`, radius 8px, padding 8px 12px. Hover: `#F4F4F5`, text `#18181B`.

### Form Inputs & Selects
- Height 36px, radius 8px, background `#FFFFFF`, border 1px solid `#E4E4E7`, horizontal padding 12px. Text `#18181B`, placeholder `#A1A1AA`.
- Focus state: Border color `#4338CA` with an exterior ring `0 0 0 1px #4338CA` (no fuzzy spread rings).

### Cards & Result Panels
- Background `#FFFFFF`, border 1px solid `#E4E4E7`, radius 12px or 14px, internal padding 16px or 24px. Header rows feature a hairline 1px divider `#F4F4F5` when separating statistical summaries from data tables.

### Upload & Dropzone Container
- Radius 16px, background `#FFFFFF`, border 1px dashed `#D4D4D8`. Hover state converts border to 1px dashed `#4338CA` with a subtle surface wash of `#EEF2FF`. Drag-and-drop iconography rendered in `#71717A`.

### Badges, Chips & Status Pills
- **Neutral Class Badge:** Height 22px, radius 9999px, background `#F4F4F5`, text `#71717A`, padding 2px 8px, typography `label-sm`.
- **Top-1 Prediction Pill:** Background `#EEF2FF`, border 1px solid `#C7D2FE`, text `#3730A3`, typography `label-sm` with tabular font features.
- **Confidence Metric Pill:** Emerald variant `#F0FDF4`, border 1px solid `#BBF7D0`, text `#15803D`.

### Checkboxes & Radios
- Size 16px by 16px, radius 4px (checkbox) or circular (radio). Border 1px solid `#D4D4D8`. Checked background `#4338CA` with `#FFFFFF` glyph, checked border `#4338CA`.

### Confidence Distribution Bar (Domain-Specific)
- Background track `#F4F4F5`, height 6px, radius 9999px. Progress fill utilizes `#4338CA` for top-ranked predictions, scaling down to `#D4D4D8` for lower rank candidates in the top-5 CIFAR-100 list.