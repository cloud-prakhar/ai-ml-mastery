# Image prompt — IoU and Non-Maximum Suppression

**Purpose:** show how overlapping detection boxes are scored by overlap and reduced to one box per object
**Audience:** 🟡 intermediate
**Aspect ratio:** 16:9 (documentation) / 9:16 (learning card)
**Teaches:** [09 Computer Vision — Classification, Detection, Segmentation and Pose](../../09-computer-vision/03-classification-detection-segmentation-and-pose.md)

## Prompt

A friendly, modern, flat-illustration diagram explaining intersection over union and non-maximum suppression.
Style: clean vector, high contrast, generous whitespace, rounded shapes,
soft shadows, educational infographic aesthetic, minimal text.
Palette: blue #2563eb, amber #d97706, purple #a21caf, green #059669, red #dc2626 on a light background.
Show three panels left to right. Left: two overlapping rectangles, one blue and one amber; their overlap filled
purple, and the combined outline traced in green, with a simple fraction "overlap over union" beneath.
Centre: a simple cartoon dog surrounded by four overlapping rectangles of different thickness, each with a small
score tag. Right: the same dog with only the thickest rectangle kept in green, and the other three shown faded
with a red cross. Connect the panels with clear arrows. Label each element with two or three words maximum.
No photorealism. No clutter. No small print. No fake logos or brand marks.

## Text overlay (keep it this short)
- Title: One Box Each
- Labels: "Overlap", "Union", "Many guesses", "Best kept"

## Accuracy checks before use
- [ ] The kept box is the one with the highest score tag
- [ ] The overlap region is the intersection, not the whole of either box
- [ ] Any text in the image is spelled correctly
- [ ] Nothing implies endorsement by a real company
