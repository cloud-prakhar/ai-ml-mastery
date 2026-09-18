# Assignments — 08 Deep Learning

Four assignments. Each produces a network you can explain: what it assumes, why it trains, and what happened when
it did not.

Rules for all four:

- **PyTorch**, installed from `requirements-dl.txt` ([module overview](../08-deep-learning/README.md)).
- **Seed everything** and record library versions; state which results you checked across seeds.
- Every network is compared with **a non-neural baseline** — logistic regression or gradient boosting — evaluated
  identically ([07 Model Evaluation](../07-model-evaluation/04-learning-curves-and-baselines.md)).
- Log the training loss, validation metric and **per-layer gradient norms**; include the plots.
- Report at least one run that failed, and what the numbers showed before you understood why.

---

## Assignment 1 — Extend the autograd engine 🟡

**Covers** Topics [1](../08-deep-learning/01-neurons-perceptrons-and-layers.md) and
[2](../08-deep-learning/02-loss-functions-computational-graphs-and-autograd.md).

**Requirements**

1. Extend the `Value` engine from Topic 2 with subtraction, division, powers, `exp`, `log` and ReLU.
2. Write a gradient check comparing every operation against finite differences and against PyTorch.
3. Build a multilayer perceptron from `Value` objects and train it on a two-dimensional problem a single neuron cannot
   solve.
4. Show, with a test, what goes wrong when `+=` is replaced by `=` in one backward rule.

**Done when** your tests pass and your network reaches the same accuracy as an equivalent PyTorch model.

---

## Assignment 2 — A training-diagnostics report 🟡

**Covers** Topics [3](../08-deep-learning/03-training-loop-batches-and-initialisation.md),
[4](../08-deep-learning/04-vanishing-gradients-normalisation-dropout-and-residuals.md) and
[5](../08-deep-learning/05-activation-functions.md).

Use a public tabular or image dataset with at least 5,000 rows.

**Requirements**

1. A learning-rate sweep on a log scale, identifying each failure region from the loss curves.
2. Batch sizes of 16, 128 and the full dataset, compared by steps and by wall-clock training time.
3. A 20-layer network trained with three initialisations and two activations; plot per-layer gradient norms at step 0
   and step 500.
4. The same deep network with batch norm, layer norm, residual connections and both; report which trained.
5. Deliberately kill ReLU units with a learning-rate spike, measure the dead share, and show one prevention working.

**Done when** someone else could diagnose a stuck training run from your report's plots alone.

---

## Assignment 3 — Architecture matches data 🔴

**Covers** Topic [6](../08-deep-learning/06-cnns-rnns-and-sequence-models.md).

**Requirements**

1. On an image dataset (for example MNIST or Fashion-MNIST, with its licence recorded), compare an MLP and a CNN with
   similar parameter counts on original, shifted and rotated test images. Then train the MLP with augmentation and
   report how much of the gap it closes.
2. Design a sequence task in which the answer depends on an early element, and compare a plain RNN, a GRU and an LSTM
   with default and raised forget-gate bias across sequence lengths. Report where each stops learning.
3. Measure the gradient reaching the first time step before and after training.

**Done when** you can state, with numbers, which built-in assumption helped on which task — and one case where it did not.

---

## Assignment 4 — Generative models, honestly evaluated 🔴

**Covers** Topics [7](../08-deep-learning/07-autoencoders-vaes-and-gans.md) and
[8](../08-deep-learning/08-other-architectures.md).

**Requirements**

1. Train a VAE and a GAN on the same small image dataset, and a tiny diffusion model on the same data or a
   low-dimensional version of it.
2. Evaluate each for **quality and diversity separately**: coverage of every class, near-duplicates of training
   examples, and a judge classifier calibrated against noise.
3. Report every GAN run across at least five seeds, including collapsed ones.
4. Use the autoencoder as an anomaly detector on held-out classes, choosing the threshold from a stated cost.
5. Check generated samples for near-copies of training images, and write a paragraph on what that would mean if the
   data were personal.

**Done when** your report would stop someone from deploying a generator on the strength of five good-looking samples.

---

[🏠 Module Home](../08-deep-learning/README.md) · [Quiz →](../quizzes/08-deep-learning.md)
