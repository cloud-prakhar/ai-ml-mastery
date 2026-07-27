<!-- status: backlog | maintained by scripts/generate_module_readmes.py -->

# 08. Deep Learning

**Level:** 🟡 Intermediate &nbsp;|&nbsp; **Effort:** Detailed module &nbsp;|&nbsp; **Status:** 📋 Backlog

Neural networks from the single perceptron to modern architectures, taught with PyTorch as the primary framework.

---

## 🎯 Learning Objectives

By the end of this module you will be able to:

- Implement forward propagation and backpropagation by hand in NumPy.
- Train, debug and regularise a network in PyTorch.
- Choose an activation, initialisation and normalisation scheme deliberately.
- Recognise vanishing and exploding gradients from training curves.

## 📚 Prerequisites

- [`02-mathematics-for-ai`](../02-mathematics-for-ai/README.md)
- [`05-machine-learning`](../05-machine-learning/README.md)

## 🗺️ Planned Topics

- Biological vs artificial neuron; the perceptron; layers, weights, biases
- Forward propagation, loss functions, backpropagation, computational graphs
- Epochs, batches, learning rate, weight initialisation
- Vanishing and exploding gradients; batch and layer normalisation; dropout; residuals
- Activations: sigmoid, tanh, ReLU, leaky/parametric ReLU, ELU, GELU, Swish, softmax
- Architectures: FFN/MLP, CNN, RNN, LSTM, GRU, autoencoders, VAEs, GANs
- Architectures: transformers, diffusion models, GNNs, Siamese, capsule, Mixture-of-Experts

## 📦 Definition of Done

This module is complete when all of the following exist and pass
[`CONTENT_CHECKLIST.md`](../CONTENT_CHECKLIST.md):

- [ ] `README.md` rewritten as the module overview with navigation links
- [ ] One topic file per planned topic, following [`templates/MODULE_TEMPLATE.md`](../templates/MODULE_TEMPLATE.md)
- [ ] Runnable code examples with pinned dependencies and expected output
- [ ] At least one Mermaid diagram per major concept
- [ ] Exercises in `../assignments/` and quiz in `../quizzes/` with separate answers
- [ ] Official references with verification dates
- [ ] Glossary and changelog updated


### Authoring notes

- PyTorch is primary; TensorFlow/Keras appear only as comparison notes.

## 🔗 Navigation

[← Previous](../07-model-evaluation/README.md) &nbsp;|&nbsp; [🏠 Repository Home](../README.md) &nbsp;|&nbsp; [Next →](../09-computer-vision/README.md)
