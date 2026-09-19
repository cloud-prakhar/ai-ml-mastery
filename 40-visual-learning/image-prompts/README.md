# Image Prompts

Prompts for generating colourful infographic-style illustrations of AI concepts — the pictures that
Mermaid cannot draw, suitable for slides, social-media learning cards and module headers.

Each prompt is a Markdown file following the template in
[`../../templates/DIAGRAM_TEMPLATE.md`](../../templates/DIAGRAM_TEMPLATE.md#image-prompt-template).

## Visual style (keep it consistent)

Friendly · colourful · modern · clean · educational · fun · high contrast · minimal text ·
beginner-friendly · flat vector illustration, no photorealism.

**Palette:** blue `#2563eb` · amber `#d97706` · purple `#a21caf` · green `#059669` · red `#dc2626`
on a light background.

## Rules

- **Accuracy first.** A beautiful diagram with the arrows the wrong way round teaches the wrong
  thing to more people, faster.
- Two or three words per label, maximum. Generated images render long text badly.
- Never generate anything implying endorsement by a real company, and no invented brand marks.
- Check spelling in the generated image before use — image models routinely mangle text.
- Provide both a 16:9 version for documentation and a 9:16 version for learning cards.

## Available prompts

| Prompt | Concept | Module |
| --- | --- | --- |
| [Convolution and Feature Maps](convolution-feature-map.md) | A filter sliding over an image | 09 |
| [IoU and Non-Maximum Suppression](iou-and-non-maximum-suppression.md) | Scoring and de-duplicating detections | 09 |
| [U-Net Segmentation](u-net-segmentation.md) | Encoder, decoder and skip connections | 09 |
| [Vision Transformer Patches](vision-transformer-patches.md) | Image patches as tokens | 09 |
| [CLIP Contrastive Learning](clip-contrastive-learning.md) | Matching images and captions | 09 |

Earlier modules, 00–08, do not have prompts yet; that gap is recorded in
[`IMPLEMENTATION_TRACKER.md`](../../IMPLEMENTATION_TRACKER.md).

---

[🏠 Repository Home](../../README.md) · [Visual Learning module](../README.md)
