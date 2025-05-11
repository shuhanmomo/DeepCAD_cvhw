# Introduction

Modern deep generative models often encode 3D shapes as high-dimensional latent vectors, which are powerful but difficult to interpret or manipulate [[arxiv.org](https://arxiv.org/abs/2304.10320#:~:text=These%20techniques%20allow%20users%20to,art%20report%2C%20we%20summarize%20research)]. This lack of interpretability limits tasks requiring explicit control or editing of generated geometry. In contrast, symbolic Computer-Aided Design (CAD) approaches represent shapes via explicit sequences of modeling operations (e.g., sketch, extrude), endowing the output with clear semantics and editability [[arxiv.org](https://arxiv.org/abs/2304.10320#:~:text=,However)]. For example, DeepCAD [citation] uses a bidirectional transformer to generate CAD command sequences, demonstrating the feasibility of learned procedural programs. However, DeepCAD and similar models are typically trained only on CAD data and lack direct conditioning on visual inputs. To address this, this project propose aligning a pretrained vision model to a symbolic CAD generator via a minimal adaptor.

Specifically, this project hypothesize that a small multilayer-perceptron (MLP) can map CLIP image embeddings into DeepCAD’s latent program space. The CLIP ViT-B/32 encoder is frozen to extract a semantic image embedding, and use a lightweight MLP adaptor to map it to the 256-dimensional latent code expected by the DeepCAD decoder. The DeepCAD decoder (also frozen) then autoregressively generates a sequence of CAD tokens. Only the adaptor’s parameters are learned during training, which allows us to leverage the rich features of the pretrained models with minimal computation. This design is inspired by recent vision-language works (e.g., BLIP-2 [[arxiv.org](https://arxiv.org/abs/2301.12597#:~:text=%3E%20Abstract%3AThe%20cost%20of%20vision,art)] ) that connect frozen encoders with small trainable modules. If successful, the proposed pipeline (CLIP → MLP adaptor → DeepCAD) will enable images to directly drive symbolic 3D program generation, laying the groundwork for user-interpretable shape synthesis from images.

# Related Works

**Neural-Symbolic CAD Generation:** Prior work has demonstrated the value of combining deep learning with symbolic CAD programs. Wu *et al.* introduced **DeepCAD**, using a bidirectional Transformer to model sequences of explicit CAD commands (sketch, extrude, etc.) and thereby generate editable 3D models. Jones *et al.*’s **ShapeAssembly** further represents shapes (e.g., furniture) as programs in a domain-specific part-assembly language. More recently, Li *et al.* proposed **Mamba-CAD**, which replaces the Transformer with a continuous state-space model while preserving the sequential program view. These neural-symbolic systems show that complex 3D geometry can be expressed as explicit procedural code learned from data, making them inherently more interpretable than raw latent representations.

**Cross-Modal Alignment:** A line of work in vision-language integration motivates our approach of using a frozen image encoder with a lightweight adaptor. CLIP provides a pretrained image encoder whose features capture rich semantic content. The *LiT* method[arxiv.org](https://arxiv.org/abs/2111.07991#:~:text=%3E%20Abstract%3AThis%20paper%20presents%20contrastive,training%20methods%20%28supervised%20and%20unsupervised) found that freezing the image tower and training only a linear text projector suffices for strong zero-shot transfer. Similarly, **BLIP-2**[arxiv.org](https://arxiv.org/abs/2301.12597#:~:text=%3E%20Abstract%3AThe%20cost%20of%20vision,art) connects a frozen vision encoder and a frozen large language model via a small “Query Transformer,” achieving state-of-the-art vision-language performance with minimal trainable parameters. Likewise, **LLaVA**[arxiv.org](https://arxiv.org/abs/2304.08485#:~:text=present%20the%20first%20attempt%20to,4%20on%20a) shows that linking a frozen vision encoder to an LLM (via instruction tuning) yields a powerful multimodal assistant. These works collectively demonstrate that pretrained encoders can be aligned to different output domains using very limited adaptation[arxiv.org](https://arxiv.org/abs/2111.07991#:~:text=%3E%20Abstract%3AThis%20paper%20presents%20contrastive,training%20methods%20%28supervised%20and%20unsupervised)[arxiv.org](https://arxiv.org/abs/2301.12597#:~:text=%3E%20Abstract%3AThe%20cost%20of%20vision,art), suggesting feasibility of our lightweight adaptor into DeepCAD.

**Generative Latent Manipulation:** Conditional generative models illustrate how external information can steer a generator’s latent space. For example, Mirza & Osindero’s *Conditional GAN*[arxiv.org](https://arxiv.org/abs/1411.1784#:~:text=,not%20part%20of%20training%20labels) concatenates a conditioning vector (such as a class label) to both the generator and discriminator, enabling control over generated content. This model shows that feeding auxiliary inputs into a network’s latent code can direct its outputs. In our context, mapping an image embedding into DeepCAD’s latent space is analogous: the visual embedding acts as a condition that guides the CAD program generation. Such conditioning of latent codes – whether via conditional GANs or similar architectures – underpins many modern generative systems and informs our strategy of integrating images with symbolic 3D generation[arxiv.org](https://arxiv.org/abs/1411.1784#:~:text=,not%20part%20of%20training%20labels).

# Methods

## Overview

## Data Processing

The ABC CAD dataset (train: 161,240; val: 8,946; test: 8,052)  [ cite ] is sub‑sampled by random sampling to 6,000 training, 1,000 validation, and 500 test models to fit a single GPU (3060 Ti) memory.  Each CAD model is rendered at 24 equally spaced azimuth angles over a fixed elevation.  A frozen CLIP ViT‑B/32 image encoder (output dim = 512) processes each view to produce per‑view feature vectors; these are aggregated via mean pooling into a single 512‑dim feature *h* for each shape【7】.  Ground‑truth latent codes *z* (256‑dim) are obtained by passing each CAD model through the pretrained DeepCAD encoder (frozen)【4】.

## Model Architecture

The work employs three frozen modules: (1) CLIP ViT‑B/32 for image embedding, (2) DeepCAD encoder for latent extraction, and (3) DeepCAD decoder for CAD command generation, with autoencoder backbone drawn from Wu *et al.*’s DeepCAD framework【4】.  Two mapping strategies are explored:

- **MLP Adaptor.** A lightweight multilayer perceptron (2–4 layers, hidden size 512) maps the pooled CLIP feature *h* to a 256‑dim latent code *ẑ*. Only adaptor weights are trained.
- **Conditional Latent GAN.** A generator *G* (MLP taking *h* plus a 64‑dim noise vector) outputs *ẑ = G(h, η)*, and a discriminator *D* (MLP conditioned on *h*) distinguishes real from generated latents.

## Loss Functions

- **MLP Adaptor:** Optimization minimizes the L2 distance between *ẑ* and ground‑truth *z*:

$$
⁍.
$$

- **Conditional Latent GAN:** Training uses the WGAN‑GP objective (with gradient penalty) [cite] to align the generated latent distribution to the real latent distribution. Optionally, a weighted L2 term is added to stabilize mapping quality.

# Experiments

## Baselines and Methods

Four approaches are evaluated to assess mapping efficacy:

- **Baseline 1: CLIP Retrieval.** Retrieve the nearest neighbor in CLIP feature space (by cosine similarity) among training samples; use its *z* as the prediction. This measures implicit alignment between CLIP and DeepCAD latents.
- **Baseline 2: PCA Regression.** Apply PCA to all training latents *z*, retain the top *k* components (e.g. *k*=100), and train a linear regressor to predict these coefficients from *h*. Reconstruct *ẑ* via the PCA decoder.
- **Method 1: MLP Adaptor.** As above, train an MLP to regress *h*→*ẑ* using L2 loss.
- **Method 2: Latent GAN.** Train *G*/*D* under WGAN‑GP to generate *ẑ* conditioned on *h*, enforcing distributional alignment in the latent space.

## **Evaluation Metrics**

The same metrics as in DeepCAD are adopted【4】:

1. **Command Accuracy:** Fraction of CAD commands with correct operation type. [math]
2. **Parameter Accuracy:** Fraction of numeric parameters within tolerance of ground truth. [math]
3. **Chamfer Distance:** Bi‑directional Chamfer distance between reconstructed and reference meshes. [math]
4. **Invalid Ratio:** Percentage of generated sequences failing to decode into valid geometry. Additionally, the average latent reconstruction error (L2 between *z* and *ẑ*) is reported.   [math]

## Latent Space Analysis

For each method on the test set, ground‑truth latents and predicted latents are projected into 2D via PCA or t‑SNE.  Plots visualize clustering and alignment.  Metrics such as average L2 error, Maximum Mean Discrepancy (MMD), and Pearson correlation between *z* and *ẑ* components are computed to quantify distributional similarity.

## Ablation Study

Two key factors are varied:

- **MLP Capacity:** Hidden layer size in the adaptor (256 vs. 512 vs. 1,024)  and number of layers to assess the trade‑off between model complexity and mapping accuracy.
- **View Aggregation:** Single‑view input (one CLIP feature for one canonical angle) versus pooled multi‑view input (24‑view mean) to evaluate the benefit of pooled multi‑view information.

Each ablation is evaluated across all metrics to identify optimal configurations for image‑conditioned symbolic 3D generation.

# Conclusion