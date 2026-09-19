# Answers — 09 Computer Vision

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../09-computer-vision.md)

---

## Images as tensors and preprocessing

**1. Shapes** — NumPy and image libraries use height-width-channels: (427, 640, 3). PyTorch uses NCHW: (1, 3, 427, 640).
`permute(2, 0, 1)` converts one to the other, and `unsqueeze(0)` adds the batch axis.

**2. 1080p memory** — 1,920 × 1,080 × 3 = 6,220,800 bytes as `uint8`, about 5.9 MiB; four times that as `float32`,
about 23.7 MiB. A batch of 32 such frames in `float32` is roughly 760 MiB before any model runs.

**3. Overflow** — `[44]`. `uint8` holds 0–255, and NumPy keeps the input type, so 300 wraps round to 300 − 256 = 44
with no warning. Convert to a wider type or float first and clip only when saving.

**4. Weighted grayscale** — the eye is far more sensitive to green than to blue. With a plain average, pure red, green
and blue all score 85; with the BT.601 weights green is 149.7 and blue 29.1, which matches how bright they look.

**5. Shadow** — R, G, B and value all fall; hue (33°) and saturation stay the same. That is why colour-based tracking
uses hue.

**6. Aliasing** — 16 pixels can represent at most 8 light–dark cycles (half a cycle per pixel). The image had 12;
sampling folds the excess back as 16 − 12 = 4 false cycles. Antialiasing blurs away the unrepresentable detail before
sampling, giving near-uniform grey.

**7. Making it square** — squash (distorts every object's shape); centre-crop (can cut off the object); letterbox —
pad with a border (wastes pixels on padding, and the model must be trained with the same padding). Whichever you pick,
use it in training and serving.

**8. Same model, three scores** — only the preprocessing: normalised as in training (0.985), not normalised (0.844),
and inverted to dark ink on white (0.131). No error was raised in any case, which is why preprocessing must be one
shared, versioned function.

**9. Decompression bomb** — a small compressed file that declares a huge image, so decoding it exhausts memory. Refuse
it from the header, before decoding — Pillow checks the declared size against `Image.MAX_IMAGE_PIXELS` when the file is
opened. Set your own limit and treat the warning band as a refusal too.

**10. EXIF** — it can contain GPS coordinates, timestamps and device details, which can reveal where a user lives.
Apply the orientation tag first (`ImageOps.exif_transpose`), or portrait photos end up sideways, then re-encode from
pixels only.

## Convolution, pooling and augmentation

**11. Cross-correlation** — true convolution flips the kernel before sliding it; libraries do not. For learned kernels
it makes no difference, because the network simply learns the flipped version. It matters when you copy a
hand-designed filter from signal-processing material: its response comes out mirrored.

**12. Output size** — $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1 = \lfloor (56 + 2 - 2 - 1)/2 \rfloor + 1 = 27 + 1 = 28$.

**13. Same padding** — $p = k // 2 = 2$: $(n + 4 - 4 - 1)/1 + 1 = n$.

**14. 27 instead of 36** — zero padding added a border of zeros above the image; at the top row the filter's upper row
covered padding instead of bright pixels, so the weighted sum was smaller. Zero padding distorts values at the edges;
reflect padding is one alternative.

**15. Receptive field** — the input region that can influence one output value. Each 3×3 layer at stride 1 adds 2
pixels: 1 + 2 + 2 + 2 = 7.

**16. VGG** — two 3×3 layers see as much as one 5×5 (5 pixels) with 18 weights instead of 25 per channel pair; three
see as much as a 7×7 with 27 instead of 49. And there is a non-linearity between each layer, so the stack can express
more than one large linear filter.

**17. Downsampling** — each layer adds $(k - 1)$ multiplied by the product of all earlier strides. After a stride-2 layer,
every later layer's contribution doubles. In the example the same three 3×3 layers reached 15 pixels with two
stride-2 layers instead of 7.

**18. Global average pooling** — it reduces each feature map to one number whatever its size, so the final linear layer
always receives one value per channel. A network that flattens its last map has its input size fixed by the linear
layer.

**19. Flips** — a mirrored digit is often not the same digit (a 2, a 3, a 7), so flipping trains the model on wrong
labels; accuracy fell on clean and shifted test sets. It would also be wrong for text recognition and for road signs with
arrows (left becomes right), and for anatomy with a meaningful left and right side.

**20. Training only** — validation and test data must look like production data, or the scores are not estimates of
production performance. The deliberate exception is **test-time augmentation**: averaging predictions over several
transformed copies of each input, for more accuracy at several times the inference cost.

## Classification, detection, segmentation and pose

**21. Task outputs** — classification: a class or a score per class, scored by accuracy, top-k or F1. Detection: a box,
class and confidence per object, scored by mAP. Semantic segmentation: a class per pixel, scored by mean IoU or Dice.
Pose: a position per keypoint, scored by PCK or OKS-based AP.

**22. IoU** — the overlap is (5, 5)–(10, 10): 5 × 5 = 25. Union: 100 + 100 − 25 = 175. IoU = 25/175 ≈ 0.143.

**23. Too-large box** — IoU divides by the union. The box has area 150 × 150 = 22,500; the object 10,000, all inside
it. IoU = 10,000 / 22,500 = 0.444. IoU penalises boxes that are too large as well as misplaced ones.

**24. 0.25** — a box stored as `xywh` (x, y, width, height) was read as `xyxy` (two corners), so (50, 50, 100, 100)
became a box ending at (100, 100) instead of (150, 150). Convert formats once, at data loading, and assert the format
in tests.

**25. Greedy NMS** — sort boxes by score; take the highest as a detection; delete every remaining box whose IoU with it
exceeds the threshold; repeat with what remains. Apply it per class.

**26. Merged people** — their boxes overlapped with IoU 0.429, above 0.3, so the lower-scoring person was deleted as a
duplicate of the higher one. Raise the threshold (0.5 kept both in the example), tune it on validation data with crowded
scenes, or use a detector that does not need NMS.

**27. Average precision** — rank all detections by confidence; mark each a hit if its IoU with a not-yet-matched true
object reaches the threshold; compute precision and recall down the ranking; AP is the area under the (interpolated)
precision–recall curve. A second box on an already-found object matches nothing new, so it is a false positive — this
is what punishes duplicates.

**28. AP50 versus COCO AP** — AP50 uses IoU 0.5 only; COCO AP averages over thresholds 0.50, 0.55, …, 0.95, rewarding
precise boxes. Several of the detections overlapped their objects by more than 0.5 but less than 0.75, and fewer still
passed the stricter thresholds, so the average fell to 0.601.

**29. Segmentation kinds** — semantic: a class for every pixel, with no notion of separate objects. Instance: a separate
mask per object. Panoptic: every pixel gets a class, and pixels of countable objects also get an instance ID, while
background "stuff" such as road and sky gets only a class.

**30. Empty segmenter** — objects covered only 5.7% of the pixels, so predicting background everywhere was right 94.3%
of the time. IoU and Dice for the object class are both 0, and per-object recall is 0.

**31. Mask metrics** — $\text{IoU} = |P \cap T| / |P \cup T|$; $\text{Dice} = 2|P \cap T| / (|P| + |T|)$. Dice is
always at least as large as IoU (Dice = 2·IoU / (1 + IoU)); both are 1 for a perfect mask and 0 for no overlap.

**32. Undercounting** — where objects touch or overlap, their pixels join into one connected region, and a semantic
mask has no way to separate them. 310 squares were placed but even the perfect mask had only 282 regions.

**33. Heatmap decoding** — argmax returns a whole heatmap cell, so the answer is rounded to the heatmap grid; each cell
was 8 image pixels wide, giving a 4-pixel error. Soft-argmax takes a probability-weighted average of all cell positions,
recovering sub-cell precision (1.3 pixels in the example) and staying differentiable.

**34. Top-down versus bottom-up** — top-down detects each person, then estimates one pose per box: accurate, but its
cost grows with the number of people. Bottom-up finds every keypoint in the image at once, then groups them into
people: roughly constant cost, but the grouping is hard in crowds.

## Faces, text, captions and restoration

**35. Face pipeline** — detect the face, align it to a standard position, embed it into a vector, and compare
embeddings with cosine similarity against a threshold.

**36. 1:1 versus 1:N** — verification checks a claimed identity with one comparison (phone unlock, passport gates).
Identification searches a gallery of N people; with N comparisons, the chance that some impostor exceeds the threshold
grows with N, so it needs stricter thresholds and usually human review.

**37. Open-set** — the system must recognise people it never saw in training, and reject people who are not enrolled.
A classifier has a fixed output per training identity and no principled way to say "none of these"; it would have to be
retrained for every new person.

**38. 1% versus 36.3%** — the embedding came from a classifier trained on identities 0–6. Its features only needed to
separate those identities, so the unseen identities landed in overlapping regions of the embedding space. The threshold
was correctly set for the data it was set on; that data did not represent new people.

**39. Metric-learning losses** — they train the embedding directly for distances. Triplet loss pulls an anchor towards a
same-identity example and pushes it away from a different identity by a margin; ArcFace adds an angular margin between
identities on the unit sphere. Trained on very many identities, the embedding generalises to new faces.

**40. NIST demographic effects** — false positive rates differed across demographic groups (by age, sex and ethnicity),
by large factors for many algorithms, and the size of the differences varied widely between algorithms. A deployer
should measure false accept and false reject rates per group on a population like the real users, choose the algorithm
and threshold accordingly, and keep a person in the loop where a match has consequences.

**41. Touching digits** — segmentation cut the line at blank columns; with no gap between characters there were no blank
columns, so each line became one piece too wide to be a character. Handwriting and joined scripts have the same
problem, which is why modern OCR reads whole lines.

**42. Compounding** — $0.91^4 \approx 0.686$, matching the 68% measured. $0.99^{12} \approx 0.886$: even 99% per
character reads a twelve-character account number wrongly about one time in nine. Hence checksums and validation rules.

**43. CTC** — connectionist temporal classification lets a model output one prediction per feature column along the
line (including a "blank"), and trains it from the line's text alone by summing over all alignments of columns to
characters. No character positions are needed in the labels.

**44. OCR and language models** — text printed in an image becomes input to the model, and instructions hidden in a
photo or scanned document can act as a prompt injection. Treat extracted text as untrusted data, never as commands.

**45. Captioning and VQA** — an image encoder (CNN or ViT) produces feature vectors; a projection maps them into the
language model's embedding space; the language model generates text attending to them, one token at a time. Their
characteristic failure is object hallucination — describing things typical of the scene that are not in the image.

**46. PSNR** — $10 \log_{10}(\text{MAX}^2 / \text{MSE})$, in decibels. Every +3 dB halves the mean squared error, because
$10 \log_{10} 2 \approx 3.01$.

**47. Learned prior** — the network learned what digits look like, so it removed what did not fit and kept the strokes;
the blur removed noise and strokes alike (+3.4 dB versus +0.7 dB). The same prior makes a restoration network fill in
detail that is typical of its training data rather than present in the scene — plausible but invented, which is
unacceptable as evidence for identification or diagnosis.

## Classic CNN architectures

**48. Landmarks** — AlexNet: a large CNN trained on GPUs with ReLU, dropout and augmentation. VGG: uniform stacks of
3×3 convolutions. Inception: parallel branches with cheap 1×1 bottlenecks. ResNet: residual connections for very deep
networks. MobileNet: depthwise separable convolutions. EfficientNet: compound scaling of depth, width and resolution
together. U-Net: an encoder–decoder with skip connections for per-pixel output.

**49. Meta device** — build the model under `torch.device("meta")`, which records shapes but allocates no memory and
performs no arithmetic; count parameters with `numel()`, and count MACs with forward hooks from each layer's output
shape and kernel size. Published accuracy figures can be read from torchvision's weight metadata without downloading
the weights.

**50. VGG-16** — most parameters are in the three fully connected layers at the end; most compute is in the
convolutions, which run at every spatial position.

**51. Depthwise separable** — standard: $k^2 C C'$ weights. Separable: $k^2 C$ for the depthwise part plus $C C'$ for
the pointwise part. Ratio: $(k^2 C + C C') / (k^2 C C') = 1/C' + 1/k^2 \approx 0.115$ for $k = 3$, $C' = 256$ — 12% in
the example. Fewer MACs is not always faster because depthwise layers do little arithmetic per byte read, so on many
GPUs they are limited by memory bandwidth. Measure latency on the target device.

**52. 1×1 bottleneck** — squeeze the channels (256 to 64) with a 1×1 convolution, do the expensive 3×3 on the thin
tensor, then expand back with another 1×1. It cut the block's cost to 12% of a plain 3×3 at 256 channels.

**53. U-Net skips** — the encoder downsamples for context and loses spatial detail; the decoder must output a
full-resolution mask. Skip connections bring each encoder level's detail back to the matching decoder level. They are
joined by concatenation, so the decoder's input has the upsampled channels plus the copied channels — twice as many.

**54. Training recipe** — architecture comparisons are only fair when both models get the same recipe (training length,
augmentation, regularisation). A new architecture trained with a modern recipe against an old one's original numbers
may be measuring the recipe, not the architecture.

**55. Transfer learning** — take a backbone pretrained on a large dataset, freeze its parameters, replace the head with
one sized for your classes, and train only the head (2,565 of 11 million parameters in the example); later unfreeze
the last blocks at a low learning rate if you have more data. You must also take the weights' preprocessing —
`weights.transforms()` — and check the licence of the weights and the dataset they were trained on.

## Detectors, vision transformers, SAM and CLIP

**56. Faster R-CNN** — the backbone and feature pyramid produce multi-scale feature maps; the region proposal network
scores and adjusts anchor boxes at every position and keeps the best few hundred proposals; RoI Align cuts a fixed-size
feature patch for each; a head classifies each and refines its box; NMS removes duplicates.

**57. Focal loss** — one-stage detectors evaluate tens of thousands of positions, almost all easy background, whose
many small losses swamped the few hard examples. Focal loss down-weights examples already classified confidently, so
training concentrates on the hard ones.

**58. DETR** — it predicts a fixed-size set of boxes from learned object queries, and during training matches
predictions one-to-one to true objects. An unmatched prediction must output "no object", so the model learns not to
duplicate — no anchors to design and no NMS to tune.

**59. ViT tokens** — (224/16)² = 196 patch tokens plus a class token = 197. At 448 × 448: 28² + 1 = 785 tokens, about
four times as many, so the attention matrix has about 16 times as many entries (785² ≈ 616,000 versus 38,809).

**60. 11.27 of 17.56** — PyTorch's multi-head attention applies its query, key, value and output projections inside a
function rather than calling them as separate layers, and the query-key and attention-value matrix products are not
layers at all, so forward hooks never saw them. Adding 5.58 GMACs of projections and 0.72 of score products by hand
matched the published figure.

**61. CNN versus ViT on small data** — a CNN builds in locality and weight sharing; a ViT must learn from examples that
nearby pixels belong together, and even where each patch is (position embeddings are learned). With 100 images the
built-in assumptions won by 29 points; with 1,257 the gap was 6. At large pretraining scale ViTs overtake CNNs, which
is why pretrained ViTs are strong backbones to fine-tune.

**62. SAM** — input: an image and a prompt (points or a box). Output: several candidate masks with predicted quality
scores. It does not output class labels — pair it with a detector or a CLIP-style model to know what the object is.

**63. CLIP** — an image encoder and a text encoder are trained together so that, in each batch, every image's embedding
is closest to its own caption's and vice versa: a symmetric cross-entropy over the matrix of image–caption
similarities. To classify, embed one caption per class ("a photo of a dog") and choose the caption closest to the image.

**64. "Not a seven"** — the toy's text encoder averages word vectors, so "not a seven" is mostly "seven". Real CLIP-style
models have been shown to behave partly like bags of words too, handling negation, word order and relations poorly.
Never build a filter that relies on them understanding "not".

**65. Typographic attack** — text written in the image, such as a handwritten label stuck on an object, changes a
CLIP-style model's prediction, because the model also learned to read. It was documented in OpenAI's analysis of
CLIP's multimodal neurons.

## Scenarios

**66. Scratches missed** — pixel accuracy is dominated by defect-free pixels; report IoU or Dice and per-defect recall
instead. Then check train/serve preprocessing (resize method and resolution — thin scratches vanish when images are
shrunk without care), camera and lighting differences from the training data, augmentation that may have blurred
scratches away, the decision threshold, and per-slice performance by defect size and product type.

**67. Counting people** — detection (or tracking over detection) with a line-crossing rule; measure counting error per
time window, plus detection recall in crowded scenes where NMS merges people. Before building: counting may not need
identification at all, so avoid face recognition; check the current privacy and surveillance rules where the shop
operates with qualified counsel, give notice, minimise and do not retain footage beyond need, and evaluate recall across
people of different appearance.

**68. Plant app on a phone** — start from efficient pretrained backbones (MobileNetV3, EfficientNet-B0), fine-tune on
labelled plant images, and choose among them by accuracy on held-out species *and* measured latency on the target
phone — not MACs alone. Tune the input resolution as a cost dial, and consider quantisation
([32 Model Optimization](../../32-model-optimization/README.md)).

**69. Day to night** — monitor per-image output statistics (detections per image, confidence distribution, box sizes,
share of empty images) and input statistics (brightness per channel), and evaluate a labelled night-time sample. Add
night images to training, add brightness and noise augmentation, and report accuracy separately for day and night.

**70. Zero-shot "unsafe" filter** — CLIP-style models handle negation poorly ("not safe" may behave like "safe"), can be
steered by text written into images, and inherit web-data biases. Build a labelled evaluation set that reflects the
policy, measure false positives and negatives per category and per group of people depicted, use the embedding as a
feature for a trained classifier rather than a single prompt, version the prompts and thresholds, and keep human review
for consequential decisions.

---

[← Back to the questions](../09-computer-vision.md) · [🏠 Module Home](../../09-computer-vision/README.md)
