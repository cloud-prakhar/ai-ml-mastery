# Quiz — 09 Computer Vision

**Level:** 🟡 Intermediate

Covers all six topics of [09 Computer Vision](../09-computer-vision/README.md).

Attempt every question before opening the answers.

Answers: [`answers/09-computer-vision.md`](answers/09-computer-vision.md)

---

## Images as tensors and preprocessing

1. What is the shape of one 640 × 427 RGB photo as a NumPy array, and as a PyTorch batch of one?
2. How many bytes does a 1080p RGB frame take as `uint8`, and as `float32`?
3. What does `np.array([200], dtype=np.uint8) + np.uint8(100)` produce, and why?
4. Why is grayscale computed as $0.299R + 0.587G + 0.114B$ rather than a plain average?
5. An orange object moves into shadow. Which of R, G, B, hue, saturation and value change?
6. Downsampling 12 stripes to 16 pixels with nearest-neighbour produced 4 stripes. Explain with the Nyquist limit.
7. Name three ways to make a 640 × 427 image square, and one risk of each.
8. The same trained CNN scored 0.985, 0.844 and 0.131 on the same test images. What changed between the runs?
9. What is a decompression bomb, and at what point should an upload service refuse one?
10. Why strip EXIF metadata from uploaded photos, and which tag should you apply before stripping?

## Convolution, pooling and augmentation

11. Why is what deep-learning libraries call "convolution" strictly cross-correlation, and when does the difference matter?
12. Write the output-size formula and use it for a 3×3 kernel, stride 2, padding 1, on a 56-pixel input.
13. What padding keeps the output size unchanged for a 5×5 kernel with stride 1?
14. In the padded example the top row of the feature map showed 27 instead of 36. Why?
15. What is a receptive field? Compute it for three stacked 3×3 layers with stride 1.
16. Why did VGG stack 3×3 layers instead of using 5×5 or 7×7 kernels? Give the weight counts.
17. Why does downsampling grow the receptive field faster than adding layers?
18. How does global average pooling let a network accept images of any size?
19. Why did horizontal-flip augmentation lower accuracy on digits? Name two other tasks where it would be wrong.
20. Why must augmentation be applied to training data only — and what is the deliberate exception?

## Classification, detection, segmentation and pose

21. Name the output and the usual metric of classification, detection, semantic segmentation and pose estimation.
22. Compute the IoU of boxes (0, 0, 10, 10) and (5, 5, 15, 15).
23. Why did a box that fully contains the object, but is twice as large, score only 0.444?
24. A correct box was scored 0.25. What went wrong, and how do you prevent it?
25. Describe greedy non-maximum suppression step by step.
26. At an NMS threshold of 0.3, two people standing close together became one detection. Explain, and give the fix.
27. How is average precision computed for one class? Why is a duplicate detection a false positive?
28. What is the difference between AP50 and COCO-style AP? Why did the same detections score 0.967 and 0.601?
29. Define semantic, instance and panoptic segmentation.
30. A segmenter that predicts nothing scored 94.3% pixel accuracy. Which metrics expose it?
31. Write IoU and Dice for masks. Which is always larger?
32. Why can counting objects from a semantic mask undercount? Give the numbers from the example.
33. Why does decoding a keypoint heatmap with argmax lose precision, and what does soft-argmax do instead?
34. Compare top-down and bottom-up pose estimation.

## Faces, text, captions and restoration

35. List the four steps of a face recognition pipeline.
36. What is the difference between verification (1:1) and identification (1:N)?
37. Why is face recognition an open-set problem, and why does that rule out a plain classifier?
38. A threshold accepted 1% of impostor pairs among known identities and 36.3% among new ones. Why?
39. What do triplet loss and angular-margin losses such as ArcFace train for?
40. What did NIST's demographic evaluation of face recognition find, and what should a deployer do about it?
41. Why was every line of touching digits read incorrectly by the segment-then-classify OCR?
42. At 91% per character, what fraction of four-character lines is read correctly? What about 99% over twelve characters?
43. What does the CTC loss let a line-level OCR model learn?
44. Why is OCR output a security concern when it is passed to a language model?
45. How do modern captioning and VQA systems connect an image encoder to a language model? What is their characteristic failure?
46. Write PSNR. By how much does the squared error change for every +3 dB?
47. Why did the trained denoiser beat a blur, and why is that same property a risk for forensic or medical use?

## Classic CNN architectures

48. Give the one idea each contributed: AlexNet, VGG, Inception, ResNet, MobileNet, EfficientNet, U-Net.
49. How can you count a model's parameters and MACs without downloading weights or doing any arithmetic?
50. VGG-16 has most of its parameters in which layers, and most of its compute in which?
51. Derive the cost ratio of a depthwise separable convolution to a standard one. Why is fewer MACs not always faster?
52. What does a 1×1 bottleneck do in ResNet-50?
53. Why does a U-Net need skip connections, and why is its decoder input twice as wide at each level?
54. ResNet-50 went from 76.1% to 80.9% top-1 with no architecture change. What does that imply for comparing papers?
55. Describe transfer learning with a frozen backbone. What must you take from the pretrained weights besides the parameters?

## Detectors, vision transformers, SAM and CLIP

56. Walk through Faster R-CNN from image to boxes.
57. What problem did RetinaNet's focal loss solve for one-stage detectors?
58. How does DETR remove the need for anchors and NMS?
59. How many tokens does ViT-B/16 produce for a 224 × 224 image? How does the attention cost change at 448 × 448?
60. Why did layer hooks count only 11.27 of ViT-B/16's 17.56 GMACs?
61. Why did a tiny CNN beat a tiny ViT by 29 points on 100 images? When do ViTs win?
62. What does SAM take as input and produce as output? What does it not produce?
63. Explain CLIP's training objective and how it classifies with no classifier head.
64. Why did the toy CLIP return sevens for "not a seven", and what does that mean for real CLIP-style models?
65. What is a typographic attack?

## Scenarios

66. A defect-inspection model is 99% pixel-accurate offline but misses most scratches in production. List what you check.
67. A retailer wants to count people entering a shop from a ceiling camera. Which task, which metric, and which legal and privacy checks?
68. A mobile app must classify plant species on a mid-range phone. How do you choose a backbone?
69. A detector trained on daytime traffic images is deployed at night. What do you monitor, and what would you change?
70. A team proposes zero-shot CLIP to flag "unsafe" images with the prompt "not safe for work". What do you tell them?

---

[🏠 Module Home](../09-computer-vision/README.md) · [Answers →](answers/09-computer-vision.md)
