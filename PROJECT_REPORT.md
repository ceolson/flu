This document is primarily to document my progress and conclusions from working on this project. For usage instructions, see `README.md`.

Lab: King Lab, Institue for Protein Design, University of Washington
Author: Conlan Olson
Date: Summer 2019

### Project goals
This project was intended to create a flexible model capable of generating realistic but synthetic hemagglutinin sequences that can be tuned for a variety of purposes. I approached it from a machine learning perspective. Keep in mind that I do not have a background in biology.
One of the first papers I found was [this one](https://arxiv.org/abs/1712.06148) from Killoran et al. at the University of Toronto where they used machine learning to generate and tune DNA sequences. My project followed the overall structure of theirs in that I created a generative model, a predictor model, and used the two back-to-back to tune the outputs of the generator. Another useful paper that uses machine learning for protein design is [here](https://arxiv.org/abs/1801.07130).

The rest of this report will treat the model as a modular system that requires:
1. A generator model to produce hemagglutinin sequences
2. A predictor to score how "good" a sequence is for a certain design constraint
3. True hemagglutinin sequences to learn from

### The generative model
The first type of training paradigm I tried was a GAN (generative adversarial model) ([reference](https://arxiv.org/abs/1406.2661)), but a GAN never worked for me. I used a Wasserstein GAN with the training method from [this paper](https://arxiv.org/abs/1704.00028) to try to stabilize the training. I tried a variety of architectures, including some fully connected and some convolutional and used a lot of different choices for hyperparameters. The training was always extremely slow and never produced realistic-looking sequences (and even sometimes collapsed completely into generating strings of 'A's). I ended up dropping, but there is no reason why a GAN shouldn't work for this problem. You can see what I think is my best implementation of a GAN in `generator.py` with the `--model=gan' option, but it still doesn't work.
I moved to using a VAE (variational autoencoder) ([reference](https://arxiv.org/abs/1312.6114)) which worked really well. Below are some of the options I've tried.

## VAE design and hyperparameters
# Architecture
I first tried a simple fully-connected architecture (`--model=vae_fc`). The encoder has dimensions `576*22 -> 512 -> 512 -> 256 -> latent dimension` (technically, the last dimension is `2*latent dimension` and the first half are treated as means and the second half are treated as standard deviations for the latent variables). The decoder is `100 -> 512 -> 512 -> 256 -> 576*22`. I used leaky ReLU activations. This architecture worked very well and I used it for most of the project.
However, it makes sense that a convolutional architecture (`--model=vae_conv`) would work well for this problem, so at the suggestion of Nao I tried a convolutional VAE with 3 layers (the middle two of which are residual) with a filter size of 5 throughout the whole model (see `generator.py` with the `--model=vae_conv` for details). I tried varying the number of channels in the residual layers to tweak the expressiveness of the model. I used 64 for a while but I have tried to go lower. If I want to use lower latent dimensions (see below) I bump it back up to 64. This can be changed with `--channel`. This also seems to work and is definitely a simpler model, which is nice.
At one point, I also tried a recurrent network that allowed flexible-length sequence-to-sequence encoding. It used LSTM cells. I wasn't able to train this though, but it is still implemented as `--model=vae_lstm`.

# Latent dimension
I used a latent dimension of 100 for most of the project, which worked fine. However, I was getting problems with unrealistic outputs, so (again at the suggestion of Nao) I tried reducing the latent dimension to constrain the model to be a bit more realistic. I've tried values from 2-100. With the fully-connected architecture, even 2 works ok. However, the convolutional architecture, being less expressive, can't go below around 20. This can be changed with `--latent_dimension`.

# Beta VAE
Following [this paper](https://openreview.net/forum?id=Sy2fzU9gl) which tries to "disentangle" the latent space by multiplying the KL loss by a parameter greater than 1, I set this parameter equal to 5 when I trained the VAEs. This can be changed with `--beta`.

# Training
When a model is trained for the first time, I use a simulated annealing protocol to ramp up the KL loss from 0 to beta. I generally trained models for 500 ish epochs, going back for more if it looked like they needed it. I think the big complicated fully connected VAE I used for a while probably got a total of about 1000 epochs of training.

# Batch normalization
I used batch normalization for the fully connected VAE.
