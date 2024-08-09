# Dynamic Diffusion: Implementing DDPM with Custom Beta Schedules

## Project Overview

This project, developed by **Nooruddin Noonari**, focuses on implementing a Diffusion Probabilistic Model (DDPM) with a custom beta schedule, allowing dynamic adjustment during the diffusion process. The model architecture is based on a U-Net structure, with ResNet and Attention mechanisms incorporated for enhanced learning. This project serves as an initial step towards improving and customizing Diffusion Models.

### Key Features:
- **Custom Beta Schedules:** Various beta schedules implemented, including linear, quadratic, and warmup variants.
- **Dynamic Upsampling and Downsampling:** Integrated Upsample and Downsample layers for efficient image processing.
- **Advanced Training Mechanism:** A novel training loop using MSE loss function optimized for diffusion models.

### Initial Results
The initial results of the model are shown below, representing the generated images after training. The results indicate a good starting point, with potential for further improvements.

![image](https://github.com/user-attachments/assets/1b8d7b96-2f66-4df8-9294-df335e6a89fd)


## Model Architecture

### UNet Backbone
- **ResNet Blocks:** Integrated ResNet blocks with timestep embeddings for capturing temporal features.
- **Attention Blocks:** Attention mechanisms for focusing on key image features, aiding in better image generation.

### Forward Pass
1. **Input:** An image tensor of shape `(batch_size, 3, 32, 32)`.
2. **Timestep Embedding:** Embeds the current timestep to capture the diffusion process's temporal aspect.
3. **Downsampling:** The input image is downsampled using convolutional layers.
4. **Mid Block:** ResNet and Attention blocks process the downsampled image.
5. **Upsampling:** The processed image is upsampled back to its original size.
6. **Output:** A tensor of the same shape as the input, representing the predicted noise.

### Backward Pass
- **Loss Function:** Mean Squared Error (MSE) is used as the loss function, comparing the predicted noise with the actual noise injected during the forward pass.
- **Optimization:** The optimizer used is Adam, with a learning rate of `1e-4`.

## Training Process

### Training Loop
1. **Sampling Timesteps:** For each image in the batch, a random timestep `t` is selected.
2. **Noise Injection:** Gaussian noise is added to the image based on the selected timestep.
3. **Model Prediction:** The model predicts the noise, attempting to denoise the image.
4. **Loss Calculation:** The MSE loss is calculated between the predicted and actual noise.
5. **Backward Pass:** The loss is backpropagated, and the model parameters are updated using the Adam optimizer.
6. **Checkpointing:** Model checkpoints are saved at the end of each epoch.

### Loss Function
- The loss function used is the **Mean Squared Error (MSE)**, which measures the difference between the predicted and actual noise. It is minimized during training, improving the model's denoising capability.

### Usage

To train the model:
```bash
python initial_ddpm.py

@misc{noonari2024dynamicdiffusion,
  author = {Nooruddin Noonari},
  title = {Dynamic Diffusion: Implementing DDPM with Custom Beta Schedules},
  year = {2023},
  howpublished = {\url{https://github.com/noorcs39/DynamicDiffusion_DDPM}},
}
