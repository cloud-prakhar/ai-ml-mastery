# Assignments — 09 Computer Vision

Four assignments: a coding exercise, a debugging exercise, a design exercise and a mini-project. Each produces
something you can defend with numbers: what it assumes, how it was measured, and where it fails.

Rules for all four:

- **PyTorch and torchvision**, installed from `requirements-dl.txt` ([module overview](../09-computer-vision/README.md)).
  Any other library is pinned in your own `requirements.txt`, with a sentence on why you needed it.
- **Use a public dataset whose licence permits your use**, and state the licence. No personal photos, no scraped
  images of people, no faces unless the dataset was collected with consent for that purpose.
- **Seed everything** and record library versions. State which results you checked across seeds.
- Every model is compared with **a simple baseline** evaluated identically — a pretrained backbone with a linear head,
  a classical method, or "predict the majority class".
- Report at least one result that surprised you, and what the numbers showed before you understood why.

---

## Assignment 1 — A preprocessing and augmentation audit 🟡 (coding exercise)

**Covers** Topics [1](../09-computer-vision/01-images-as-tensors-and-preprocessing.md) and
[2](../09-computer-vision/02-convolution-pooling-and-augmentation.md).

Use a public image classification dataset with at least 2,000 images and at least five classes.

**Requirements**

1. Write **one** preprocessing function — decode, orientation, size check, resize with antialiasing, channel order,
   scaling, normalisation with training-set statistics — and use it for both training and inference. Write a test that
   fails if the two paths ever produce different tensors for the same file.
2. Fine-tune a pretrained ResNet-18 head. Then evaluate the same model with four deliberately wrong serving pipelines:
   no normalisation, BGR instead of RGB, a different resize library, and no antialiasing. Report the accuracy drop for
   each.
3. Train with no augmentation, with label-preserving augmentation, and with one augmentation you argue breaks labels
   for this dataset. Evaluate each on the clean test set and on a shifted or recoloured copy.
4. Add an upload validator: a byte limit, a pixel limit enforced before decoding, an allowed-format check, and EXIF
   stripping after applying orientation — with tests for each, including a decompression-bomb test file generated in
   the test.

**Done when** your tests pass, and your report ranks the serving mistakes by how much they cost and states which
augmentations you would ship.

---

## Assignment 2 — Detection metrics from scratch, and a broken evaluator 🟡 (debugging exercise)

**Covers** Topic [3](../09-computer-vision/03-classification-detection-segmentation-and-pose.md).

**Requirements**

1. Implement IoU, greedy NMS, per-class NMS and average precision (at a single IoU threshold, and COCO-style averaged
   over 0.50–0.95) from scratch. Test IoU and NMS against `torchvision.ops`.
2. Build a small synthetic detection set — ground-truth boxes and scored detections — for which you can compute AP by
   hand, and test your implementation against the hand calculation.
3. **Debugging.** Each of the following bugs changes the reported mAP. Introduce them one at a time into your evaluator,
   predict the direction of the change before running it, then measure it:
   - boxes read as `xyxy` when they are `xywh`
   - a ground-truth object allowed to be matched more than once
   - NMS applied across classes instead of per class
   - detections not sorted by confidence before matching
   - the IoU threshold compared with `>` instead of `>=` on boxes that touch the threshold exactly
4. For each bug, write the test that would have caught it.

**Done when** every test passes on the correct evaluator, every bug is caught by at least one test, and your report
lists your predictions next to the measured effects.

---

## Assignment 3 — Choose a backbone for a latency budget 🔴 (design exercise)

**Covers** Topics [5](../09-computer-vision/05-classic-cnn-architectures.md) and
[6](../09-computer-vision/06-detectors-vit-sam-and-clip.md).

A client needs an image classifier that must answer within a stated latency on a stated device — choose one you can
actually measure on, such as your laptop's CPU, and a second, slower setting (fewer threads, or a smaller device).

**Requirements**

1. Shortlist at least four backbones from different families, including one MobileNet or EfficientNet and one ViT.
   Tabulate parameters and MACs counted on the meta device, and explain any disagreement with published figures.
2. Fine-tune each with the same recipe on the same data, and report accuracy with a confidence interval
   ([07 Model Evaluation](../07-model-evaluation/01-cross-validation-and-comparing-models.md)).
3. Measure latency on both settings at batch size 1 and at your throughput batch size, reporting the median and the
   95th percentile over repeated runs after a warm-up. Show where the MAC ranking and the latency ranking disagree.
4. Try two input resolutions for the best two candidates.
5. Write a one-page design decision: the recommended backbone and resolution, the evidence, the risks, and what would
   make you change the decision.

**Done when** someone else could reproduce every number in your table, and your recommendation follows from it.

---

## Assignment 4 — Segmentation or verification, with a responsible-use report 🔴 (mini-project)

**Covers** Topics [3](../09-computer-vision/03-classification-detection-segmentation-and-pose.md),
[4](../09-computer-vision/04-faces-text-captions-and-restoration.md) and
[5](../09-computer-vision/05-classic-cnn-architectures.md).

Choose **one**:

- **Option A — segmentation.** Train a U-Net and a pretrained-backbone segmentation model on a public segmentation
  dataset (medical, satellite, industrial or street scenes).
- **Option B — verification without faces.** Build an embedding-based verifier for a public non-face identity task —
  for example, recognising individual animals, products or handwriting writers — trained with a metric-learning loss.

**Requirements**

1. A baseline, and your model, evaluated with the right metrics: mean IoU and Dice per class with per-object recall
   (A), or false accept and false reject rates at a stated operating point on identities **disjoint from training** (B).
2. Evaluation on at least three slices you choose and justify — object size, image source, lighting, or group.
3. For A: count objects from your predicted masks and compare with the true count; explain any undercount. For B:
   compare a threshold chosen on training identities with one chosen on held-out identities.
4. A model card: intended use, out-of-scope uses, data and its licence, metrics per slice, known failure modes, and —
   if your system could be applied to people — which regulations would apply and why it would need legal review. (No
   legal advice is expected or wanted; the point is to know which questions to ask.)

**Done when** your model beats the baseline on the right metric, and your model card would stop someone from deploying
it outside the conditions you tested.

---

[🏠 Module Home](../09-computer-vision/README.md) · [Quiz →](../quizzes/09-computer-vision.md)
