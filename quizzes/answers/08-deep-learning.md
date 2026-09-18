# Answers — 08 Deep Learning

Explanations, not just answers. If you got one right for the wrong reason, that still counts as
getting it wrong.

[← Back to the questions](../08-deep-learning.md)

---

## Neurons, perceptrons and layers

**1. Differences** — artificial neurons output one number with no timing, where biological neurons spike; they learn
by one global rule (backpropagation) rather than many local mechanisms; and their "synapses" are plain weights. Also
acceptable: scale, and that a network is a differentiable function, not a model of a brain.

**2. Perceptron rule** — $w \leftarrow w + (y - \hat{y})x$, $b \leftarrow b + (y - \hat{y})$. When the prediction is
correct, $y - \hat{y} = 0$ and nothing changes.

**3. XOR** — a perceptron's decision boundary is one straight line. XOR's positive cases sit on opposite corners, so no
line separates them; the rule never converges and got all four examples wrong in every epoch from the third onwards.

**4. Parameter count** — $64 \times 32 + 32 = 2{,}080$. The weight's shape is (32, 64): (outputs, inputs), one row per neuron.

**5. Collapse** — $W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)$: a single weight matrix and bias.

**6. Universal approximation** — a network with one hidden layer and a non-linearity can approximate any continuous
function on a bounded region, given enough neurons. It does not promise that training will find those weights, that a
reasonable number of neurons suffices, or that the result generalises.

## Loss functions, computational graphs and autograd

**7. Forward and backward** — the forward pass records every operation and its inputs as a graph. The backward pass
starts with gradient 1 at the output and visits nodes in reverse topological order, each applying its local derivative
rule to pass gradients to its inputs.

**8. Local rules** — addition copies the gradient to both inputs; multiplication sends each input the other input times
the gradient; tanh multiplies the gradient by $1 - \tanh^2$.

**9. Accumulation** — a value used several times receives a gradient from each use, and the chain rule sums them. In
$y = x \cdot x + x$, $x$ receives $x$, $x$ and $1$: $2x + 1 = 7$ at $x = 3$. Assignment instead of `+=` would keep only
the last contribution.

**10. Loss choice** — regression: linear output, mean squared error (or Huber for outliers). Binary: one logit,
`BCEWithLogitsLoss`. Multi-class: one logit per class, `CrossEntropyLoss`. Multi-label: one logit per label,
`BCEWithLogitsLoss`.

**11. Squared error with sigmoid** — its gradient with respect to the logit is $2(p - y)p(1 - p)$, and $p(1-p)$ is near
zero when the sigmoid saturates, including when confidently wrong. Cross-entropy's is $p - y$. At logit −6 with target
1, the squared-error gradient was about 200 times smaller.

**12. `log(sigmoid(z))`** — in float32, `sigmoid(30)` rounds to exactly 1.0, so `log(1 - p)` is `log(0)`. Use
`BCEWithLogitsLoss` or `CrossEntropyLoss`, which combine the sigmoid or softmax and the log in one stable formula.

**13. No `zero_grad()`** — gradients add up across steps, so each update uses the sum of all previous gradients and
training becomes erratic.

**14. `torch.no_grad()`** — evaluation needs no graph; building one wastes memory and time.

## The training loop, batches and initialisation

**15. Terms** — a batch is the examples used for one update; a step is one update; an epoch is one pass over the
training data.

**16. Epochs** — equal epochs can mean very different numbers of updates. Compare runs by steps and data seen, and
revisit the learning rate when changing batch size.

**17. Shuffling** — ordered data makes consecutive batches pull the weights consistently in wrong directions.

**18. 2.3** — $\ln 10 \approx 2.30$ is the cross-entropy of guessing uniformly among 10 classes. A loss still near it
means the model has barely learned: the learning rate is too small, or no gradient is flowing.

**19. Symptoms** — too small: the loss barely moves. Slightly too large: it falls fast then plateaus high or oscillates.
Much too large: it climbs, diverges or becomes `nan`; at learning rate 10 the network ended worse than guessing.

**20. Symmetry** — identical neurons compute identical outputs, get identical gradients and stay identical. Started at
a constant, 32 hidden neurons ended as one distinct neuron and the network scored 10%.

**21. Scale** — at 0.01 the activations shrank about sixfold per layer to around $10^{-8}$; at 1.0 most tanh units
saturated at ±1, where the slope is almost zero.

**22. Xavier and He** — Xavier: $2/(n_{\text{in}} + n_{\text{out}})$, for tanh and sigmoid. He: $2/n_{\text{in}}$, for
ReLU, whose zeroing of negative inputs halves the variance each layer; the 2 compensates.

## Vanishing gradients, normalisation, dropout and residuals

**23. Sigmoid** — each layer multiplies the gradient by the sigmoid's derivative, at most 0.25; over 20 layers that is at
most $0.25^{20} \approx 10^{-12}$ before even counting the weights.

**24. The ratio** — an optimiser step that moves the last layer sensibly moves the first by essentially nothing; the
early layers stay at their random initial values.

**25. Exploding gradients** — gradient clipping, careful initialisation, normalisation, a lower learning rate. For RNNs,
gated cells.

**26. Statistics** — batch norm: each feature across the examples in the batch. Layer norm: each example across its
features.

**27. Batch dependence** — the example was normalised with its batch's mean and variance, which changed with its
neighbours. At inference, batch norm must use stored running averages, and it fails with a batch of one in training mode.

**28. Transformers** — layer norm treats each example independently, identically in training and inference, which suits
variable-length sequences, small batches and token-by-token generation.

**29. Dropout** — in training, each unit is zeroed with probability $p$ and survivors are scaled by $1/(1-p)$ so the
expected activation is unchanged; in evaluation it does nothing. Forgetting `model.eval()` makes predictions random.

**30. What changed** — the predictions became less overconfidently wrong: validation loss fell from 0.506 to 0.401
while accuracy barely moved. Its accuracy gain on this small problem was modest.

**31. Residual Jacobian** — $\partial x_{l+1}/\partial x_l = I + \partial F/\partial x_l$. The identity term passes the
gradient through unchanged, giving every layer a direct path to the loss.

**32. 30 layers** — only the residual networks learned (0.940, and 0.976 with layer norm). One run on small data does
not show that a normalised plain network can never train; with tuning it might. The point is robustness.

## Activation functions

**33. Maximum gradients** — sigmoid 0.25, tanh 1 (at zero only), ReLU exactly 1 for every positive input.

**34. Dead ReLU** — a unit whose input is negative for every training example: zero output, zero gradient, no further
learning. Usually caused by a large update, such as a learning-rate spike, pushing its bias negative.

**35. Leaky ReLU** — its negative slope of 0.01 passes 100 times less gradient; within 500 steps it revived none of the
units. Prevention — sensible learning rate, warm-up, clipping, He initialisation — beats repair.

**36. GELU and Swish** — in transformers and some image models. They are smooth and let a little gradient through for
negative inputs, with slightly negative slopes in part of that range, unlike ReLU's hard zero.

**37. Softmax** — softmax is unchanged by subtracting a constant, and subtracting the maximum keeps every exponent at most
zero, so `exp` cannot overflow; without it, logits near 1000 gave `nan`.

**38. Output activation** — none during training in both cases: regression outputs are linear, and classification losses
apply the sigmoid or softmax internally.

## MLPs, CNNs and recurrent networks

**39. Assumptions** — MLP: none. CNN: patterns are local and can appear anywhere. RNN: data arrives in order and the past
matters.

**40. Definitions** — local connectivity: each unit sees a small patch. Weight sharing: one kernel at every position.
Pooling: summarising a neighbourhood, giving tolerance to small shifts.

**41. Convolution parameters** — $(3 \times 3 \times 16 + 1) \times 32 = 4{,}640$. The same kernel slides over every
position, so the count depends on kernel size and channels, not image size.

**42. Shift** — the MLP tied evidence to fixed pixel positions. The CNN's detectors slide across the image and the global
max pool keeps each detector's strongest response wherever it occurs, so a shifted digit gives nearly the same features.

**43. RNN** — $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$ with the same weights at every step. Training unrolls the loop
into one layer per step and backpropagates through it, multiplying by $W_h$ and the tanh derivative once per step.

**44. LSTM gates** — forget: how much of the old cell state to keep; input: how much new information to write; output:
how much to expose. With the forget gate near 1, the cell state is updated mainly by addition, an additive path along
which gradients survive.

**45. Fresh LSTM** — PyTorch initialises the forget-gate bias near zero, so the gate starts around 0.5 and halves the
signal every step. Raising the forget-gate bias kept a usable gradient over 200 steps.

**46. GRU** — it merges the forget and input gates into one update gate and has no separate cell state: fewer parameters,
often similar results.

**47. Transformers versus RNNs** — attention connects every position to every other directly, with no long chain of
multiplications, and trains in parallel across positions. RNNs remain useful for streaming data and small devices.

## Autoencoders, VAEs and GANs

**48. Self-supervised** — the training target is the input itself, so no labels are needed. A linear autoencoder trained
with squared error learns the same subspace as PCA.

**49. Autoencoder versus PCA** — non-linear layers can learn a curved subspace. It gives up PCA's ordered, interpretable
components, and needs training and tuning.

**50. Poor generator** — its codes form scattered clusters with gaps; decoding a point from a gap gives nonsense.

**51. VAE loss** — reconstruction error, to encode enough information to rebuild the input, plus the KL divergence from a
standard normal, to keep code distributions overlapping and fill the space so random codes decode to plausible data.

**52. Reparameterisation** — sampling is not differentiable, so the sample is written as $z = \mu + \sigma\epsilon$ with
$\epsilon$ from a standard normal; gradients flow through $\mu$ and $\sigma$.

**53. Evaluating samples** — a classifier's confidence is weak evidence of realism: it was fairly confident about pure
noise and put half of it in one class. Generative models need purpose-built evaluation.

**54. GAN** — $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$. Two networks optimise opposing
objectives; each loss depends on the other network's current state, so neither loss decreasing is a reliable signal.

**55. Mode collapse** — the generator produces only part of the data's variety. Every run first produced only one of the
two modes; one was still stuck after 3,000 steps. Detect it by measuring the diversity of generated samples.

## Transformers, diffusion, GNNs and other architectures

**56. Self-attention** — the long chain of multiplications through time, and so the vanishing long-range gradient. It is
taught in [11 Transformers](../../11-transformers/README.md).

**57. Diffusion** — $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$; train a network to predict
$\epsilon$ from $x_t$ and $t$ with squared error.

**58. Diffusion versus GAN** — plain regression over noisy copies of all the data leaves no incentive to ignore a mode
and no adversarial instability. The cost is many network evaluations per sample — 100 here.

**59. GCN** — $H^{(l+1)} = \text{ReLU}(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2}H^{(l)}W^{(l)})$: add self-loops to the adjacency
matrix, normalise by degree so busy nodes do not dominate, mix each node's features with its neighbours', then apply
shared learned weights.

**60. GCN versus MLP** — the features barely separated the classes, but neighbours were mostly in the same community, so
message passing spread the labels along the edges. It hurts when connected nodes tend to differ (heterophily).

**61. Siamese** — a similarity function: one shared encoder maps inputs to embeddings compared by distance. It enables
one-shot recognition of classes never seen in training, search and deduplication.

**62. Mixture-of-experts** — 264,704 stored, 33,088 used per input (one of eight experts). Even an untrained router sent
inputs unevenly — 13 to one expert, 243 to another — which training compounds without a load-balancing loss.

**63. Capsules** — a principled idea still has to scale on real hardware and data to be adopted.

**64. float64** — so outputs match exactly across CPUs: rounding differences in float32 grow over hundreds of training
steps into different digits. Runs that are failing to learn wander chaotically, so they print a band instead of digits.

---

[← Back to the questions](../08-deep-learning.md) · [🏠 Module Home](../../08-deep-learning/README.md)
