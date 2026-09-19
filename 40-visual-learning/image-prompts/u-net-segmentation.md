# Image prompt — U-Net Segmentation

**Purpose:** show the U shape: downsampling for context, upsampling for detail, and skip connections between them
**Audience:** 🟡 intermediate
**Aspect ratio:** 16:9 (documentation) / 9:16 (learning card)
**Teaches:** [09 Computer Vision — Classic CNN Architectures](../../09-computer-vision/05-classic-cnn-architectures.md)

## Prompt

A friendly, modern, flat-illustration diagram explaining a U-Net image segmentation network.
Style: clean vector, high contrast, generous whitespace, rounded shapes,
soft shadows, educational infographic aesthetic, minimal text.
Palette: blue #2563eb, amber #d97706, purple #a21caf, green #059669 on a light background.
Show: a large letter U made of stacked blocks. The left arm, in blue, goes downwards, with each block narrower
in size but thicker in depth. The bottom block is purple. The right arm, in amber, goes upwards, each block
growing back to full size. Horizontal dashed arrows connect each left block to the right block at the same height.
At the top left, a simple photo of a cat on grass enters; at the top right, the same outline filled with flat
colours comes out — cat in green, grass in amber. Label each element with two or three words maximum.
No photorealism. No clutter. No small print. No fake logos or brand marks.

## Text overlay (keep it this short)
- Title: The U-Net
- Labels: "Shrink: context", "Grow: detail", "Skip", "Mask"

## Accuracy checks before use
- [ ] The output mask is the same size as the input image
- [ ] Skip arrows go from the left (encoder) to the right (decoder) at the same level
- [ ] Any text in the image is spelled correctly
- [ ] Nothing implies endorsement by a real company
