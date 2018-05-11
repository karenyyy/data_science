
# ACGAN Model
[Conditional Image Synthesis with Auxiliary Classifier GANs paper](https://arxiv.org/pdf/1610.09585.pdf)

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/acgan.jpg" width="700">

- CNN have brought advances in image classification
- CNN can also be reversed to generate images from scratch (generative models)
- One type of generative model are generative adversarial networks (GANs)
- Special type of GAN is Auxiliary Classifier GANs (ACGAN)
- GAN with class-label conditioning to generate images


## Examples (with label conditioning)

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/generated-images.png" width="800">

# Introduction to Generative Adversarial Networks (GANs)

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/GANs.png" width="800">

- GAN is composed of two competing neural network models (often CNNs)
- **Generator: ** takes noise input and generate a realistic image
- **Discriminator: ** takes real and fake images and has to distinguish the fake from the real
- Two networks play an adversarial game
- generator learns to produce more and more realistic samples
- discriminator learns to get better and better at distinguishing generated data from real data.
- networks are trained simultaneously

** Training on real image: **
- Discriminator should classify real image as real
- Ouput probability close to 1

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/Dis.png" width="800">

** Training on fake image: **
- Generator generate fake image from noise
- Discriminator should classify fake image as fake
- Ouput probability close to 0

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/gen.png" width="900">



# Auxiliary Classifier ACGAN
Proposed a new method for improved training of GANs by conditioning input with class labels.

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/acgan-2.png" width="300">

**Multi-input multi-output network:**
- **Inputs:** class embedding and noise vector
- **Outputs:** binary classifier (fake/real images) and multi-class classifier (image classes)

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/acgan-3.png" width="800">
