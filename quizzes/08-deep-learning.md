# Quiz — 08 Deep Learning

**Level:** 🟡 Intermediate

Covers all eight topics of [08 Deep Learning](../08-deep-learning/README.md).

Attempt every question before opening the answers.

Answers: [`answers/08-deep-learning.md`](answers/08-deep-learning.md)

---

## Neurons, perceptrons and layers

1. Name three ways an artificial neuron differs from a biological one.
2. Write the perceptron learning rule. What happens to the weights when a prediction is correct?
3. Why did the perceptron learn AND and OR but never XOR?
4. A layer maps 64 inputs to 32 outputs. How many parameters does it have, and what shape is `nn.Linear(64, 32).weight`?
5. Show algebraically why two stacked linear layers with no activation equal one linear layer.
6. What does the universal approximation theorem promise, and what does it not promise?

## Loss functions, computational graphs and autograd

7. What does the forward pass record, and what does the backward pass do with it?
8. Give the local backward rule for addition, multiplication and tanh.
9. Why must gradients be accumulated with `+=`? Use $y = x \cdot x + x$.
10. Which loss and output layer would you use for regression, binary classification, multi-class and multi-label problems?
11. Why does squared error with a sigmoid output train poorly when the model is confidently wrong? Give both gradients.
12. Why did computing `log(sigmoid(z))` by hand return infinity at a logit of 30, and what is the fix?
13. What happens if you forget `optimiser.zero_grad()`?
14. Why wrap evaluation in `torch.no_grad()`?

## The training loop, batches and initialisation

15. Define batch, step and epoch.
16. In 20 epochs, batch size 8 took 3,380 steps and full-batch took 20. What does that imply about comparing runs by epochs?
17. Why shuffle every epoch?
18. The loss at learning rate 0.001 stayed near 2.29. Where does 2.3 come from, and what does it tell you?
19. List the loss-curve symptoms of a learning rate that is too small, slightly too large, and much too large.
20. Why must weights not all start at the same value? Describe the result in the example.
21. What happened to activations through 10 tanh layers with weights of standard deviation 0.01, and with 1.0?
22. State the Xavier and He variances. Which suits ReLU, and why the factor of 2?

## Vanishing gradients, normalisation, dropout and residuals

23. Why does backpropagation through many sigmoid layers vanish? Give the key number.
24. The sigmoid network's last-to-first gradient ratio was around $10^{16}$. What does that mean for training?
25. Name four remedies for exploding gradients.
26. What do batch normalisation and layer normalisation each compute their statistics over?
27. Why did the same example get different batch-norm outputs in two batches, and why does that matter at inference?
28. Why do transformers use layer normalisation?
29. How does dropout behave in training and in evaluation mode, and what is the scaling by $1/(1-p)$ for?
30. Dropout barely changed validation accuracy but lowered validation loss. What changed?
31. Write the Jacobian of a residual block and explain why it helps gradients.
32. At 30 layers, which networks learned? What should you not conclude from one such run?

## Activation functions

33. Compare the maximum gradients of sigmoid, tanh and ReLU.
34. What is a dead ReLU unit, and what usually causes it?
35. After a learning-rate burst killed most units, why did leaky ReLU not revive them?
36. Where are GELU and Swish typically used, and how do they differ from ReLU?
37. Why subtract the maximum logit before computing softmax?
38. Which activation should the output layer have during training for regression and for classification?

## MLPs, CNNs and recurrent networks

39. What assumption does each of MLP, CNN and RNN build in?
40. Define local connectivity, weight sharing and pooling.
41. How many parameters does a 3×3 convolution from 16 to 32 channels have? Why is it independent of image size?
42. The MLP and CNN had similar parameter counts and similar unshifted accuracy. Why did only the CNN survive a one-pixel shift?
43. Write the RNN update and explain backpropagation through time.
44. What are the LSTM's forget, input and output gates, and why does the cell state help gradients?
45. Why did a freshly initialised LSTM lose gradient through time almost as fast as a plain RNN?
46. How do GRUs differ from LSTMs?
47. Why have transformers replaced RNNs for most sequence tasks, and where are RNNs still useful?

## Autoencoders, VAEs and GANs

48. Why is an autoencoder self-supervised? How does a linear autoencoder relate to PCA?
49. Why did the non-linear autoencoder beat PCA, and what does it give up?
50. Why is an ordinary autoencoder a poor generator?
51. Write the VAE loss and explain what each term pushes the model to do.
52. What is the reparameterisation trick, and why is it needed?
53. A classifier gave uniform noise 0.59 average confidence. What does that say about evaluating generated samples?
54. Write the GAN objective. Why is there no single loss to watch?
55. What is mode collapse, how did it show in the example, and how is it detected?

## Transformers, diffusion, GNNs and other architectures

56. What problem of RNNs does self-attention remove, and where is it taught in this repository?
57. Write the diffusion forward process and the training objective.
58. Why did diffusion cover both modes where the GAN collapsed, and what does it cost?
59. Write the GCN update and explain each part.
60. Why did the GCN reach 0.982 from four labels while the MLP was at chance? When would the graph hurt?
61. What does a Siamese network learn, and what does that enable?
62. In the mixture-of-experts example, how many parameters were stored and used per input? What problem did the routing show?
63. What lesson do capsule networks teach?
64. Why do the examples in this module use float64, and when do they print a band instead of a number?

---

[🏠 Module Home](../08-deep-learning/README.md) · [Answers →](answers/08-deep-learning.md)
