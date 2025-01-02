# Deep Convolutional GAN and Adversarial AutoEncoder for Anime Faces Generation

## Overview

This project explores the implementation of two generative models, **Deep Convolutional Generative Adversarial Networks (DCGANs)** and **Adversarial AutoEncoders (AAEs)**, to generate synthetic anime faces. By augmenting the Anime Faces dataset, the models aim to create realistic, high-quality synthetic images. Both models' performance is evaluated using **Inception Score (IS)** and **Frechet Inception Distance (FID)**.

## Objectives

1. **Data Augmentation**: Generate synthetic anime face data to enhance the original dataset.
2. **Model Comparison**: Analyze the performance of DCGAN and AAE models using key metrics.
3. **Validation**: Evaluate generated images using IS and FID scores.

## Highlights

- Implementation of **DCGAN** with ReLU activation in the generator and LeakyReLU in the discriminator.
- Development of **AAE** with convolutional encoder-decoder networks and a linear discriminator.
- Evaluation with objective metrics like **IS** and **FID** for unbiased quality assessment.
- Results show promising synthetic data generation with comparable quality to original anime faces.

## Dataset

### Source

**Anime Faces Dataset**: [Available on Kaggle](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)

- **Format**: Images are converted to NumPy arrays and structured into PyTorch DataLoader objects.
- **Preprocessing**:
  - Resized images to 128x128x3.
  - Normalized pixel values to the range [-1, 1] for compatibility with model requirements.
  - Augmented with random flips and rotations to enhance diversity during training.

## Model Implementation

### 1. **Deep Convolutional GAN (DCGAN)**

- **Architecture**:
  - **Generator**: Employs transposed convolutional layers with batch normalization and ReLU activations, ending with a Tanh activation.
  - **Discriminator**: Consists of convolutional layers with batch normalization and LeakyReLU activations (negative slope of 0.2).
- **Training Strategy**:
  - Learning rate: Initially set to 1e-3 and adjusted iteratively.
  - Batch size: Experimented with sizes ranging from 64 to 512, with 64 providing optimal results.
  - Checkpoint logic: Saved models at six stages to prevent vanishing gradients and enable progressive training.
  - Total Epochs: Trained for 400 epochs, monitoring generator and discriminator losses.
- **Key Adjustments**:
  - Changed the generator’s activation functions from LeakyReLU to ReLU for better feature learning.
  - Reduced the discriminator’s negative slope to improve learning balance.

### 2. **Adversarial AutoEncoder (AAE)**

- **Architecture**:
  - **Encoder**: A convolutional network using LeakyReLU activations.
  - **Decoder**: A mirrored architecture of the encoder, ending with Tanh activation.
  - **Discriminator**: A linear network with sigmoid activation for real vs. fake classification.
- **Training Strategy**:
  - Learning rate and negative slope tuned to balance encoder and discriminator losses.
  - Batch size reduced to minimize memory usage (9-10 GB GPU, 8-10 GB RAM).
  - Epochs: Training concluded after observing discriminator loss convergence (10 epochs).
- **Key Observations**:
  - Encoder learned significantly faster than the discriminator.
  - Loss convergence was evident by zoomed plots of discriminator loss over batches.

## Results

### Generated Images

- **DCGAN**: Over 400 epochs, generated images showed significant improvement in detail and realism.
- **AAE**: Produced reconstructed images with reduced noise, demonstrating strong feature learning.

### Evaluation Metrics

1. **Inception Score (IS)**:
   - Measures diversity and quality of generated images.
   - DCGAN and AAE achieved IS scores consistent with improved classification results.
2. **Frechet Inception Distance (FID)**:
   - Calculates the similarity between generated and real images.
   - Scores decreased over epochs, indicating improved alignment with original data.

### Sample Visuals

- Includes side-by-side comparisons of real and generated images.
- Loss convergence graphs for both models highlight training progress.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/DCGAN-AAE-AnimeFaces.git
   cd DCGAN-AAE-AnimeFaces
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:
   ```bash
   jupyter notebook DCGAN_AAE_AnimeFaces.ipynb
   ```

## Pretrained Models

Pretrained models for both DCGAN and AAE are available for further analysis or fine-tuning. Access them via:
[Google Drive Link](https://drive.google.com/drive/folders/1rlCIl5CZm_MolS-Uc797QIA7uw3UUVsH?usp=drive_link)

## Future Work

1. Extend dataset size and diversity to improve generalization.
2. Experiment with advanced generative architectures like StyleGAN or BigGAN.
3. Optimize training processes with enhanced computational resources (e.g., GPUs with higher RAM).
4. Investigate alternative evaluation metrics for a more comprehensive analysis of synthetic data quality.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or collaboration, please contact [your email/username].

