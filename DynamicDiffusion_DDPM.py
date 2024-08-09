import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
import torchvision.utils
import os

# Hyperparameters
batch_size = 32
num_epochs = 20  # Increased for better learning
total_timesteps = 1000
norm_groups = 8
initial_learning_rate = 2e-4  # Starting learning rate

img_size = 32
img_channels = 3
clip_min = -1.0
clip_max = 1.0

first_conv_channels = 128  # Increased for higher capacity
channel_multiplier = [1, 2, 4, 8]
widths = [first_conv_channels * mult for mult in channel_multiplier]
has_attention = [False, True, True, True]
num_res_blocks = 2

# Checkpoint directory
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Data loading and augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load dataset pairs (input and target images)
dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

class GaussianDiffusion:
    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000, clip_min=-1.0, clip_max=1.0):
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Cosine schedule for betas
        def cosine_beta_schedule(timesteps):
            return torch.tensor([math.cos(0.5 * math.pi * t / timesteps) ** 2 for t in range(timesteps)])

        self.betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], 0)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        
        self.posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(torch.max(self.posterior_variance, torch.tensor(1e-20)))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

    def _extract(self, a, t, x_shape):
        batch_size = x_shape[0]
        return a.to(t.device)[t].reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise):
        return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(-1, *([1] * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -self.emb)

    def forward(self, t):
        emb = t[:, None] * self.emb[None, :].to(t.device)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8, activation_fn=nn.SiLU()):
        super(ResidualBlock, self).__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = activation_fn

        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        x = self.activation(self.norm1(x))
        x = self.conv1(x)
        x = self.activation(self.norm2(x))
        x = self.conv2(x)
        return x + residual

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, num_groups=8):  # Increased number of heads for better attention
        super(AttentionBlock, self).__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(B, self.num_heads, 3 * (C // self.num_heads), H * W)
        q, k, v = qkv.split(C // self.num_heads, dim=2)

        attn = torch.einsum('bnhq,bnhk->bnhk', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bnhk,bnhv->bnhv', attn, v)
        out = out.reshape(B, C, H, W)

        return self.proj(out) + x

class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, img_channels, base_channels, widths, has_attention):
        super(UNet, self).__init__()
        self.img_channels = img_channels
        self.base_channels = base_channels
        self.widths = widths
        self.has_attention = has_attention

        self.time_embed = nn.Sequential(
            TimeEmbedding(base_channels * 4),
            nn.Linear(base_channels * 4, base_channels * 4),
            nn.SiLU(),
            nn.Linear(base_channels * 4, base_channels * 4)
        )

        self.in_conv = nn.Conv2d(img_channels * 2, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in range(len(widths)):
            in_channels = base_channels if i == 0 else widths[i-1]
            out_channels = widths[i]
            self.downs.append(nn.ModuleList([
                ResidualBlock(in_channels, out_channels, num_groups=norm_groups),
                ResidualBlock(out_channels, out_channels, num_groups=norm_groups),
                DownSample(out_channels) if i != len(widths) - 1 else nn.Identity()
            ]))
            if has_attention[i]:
                self.downs[-1].append(AttentionBlock(out_channels, num_groups=norm_groups))

        self.middle = nn.ModuleList([
            ResidualBlock(widths[-1], widths[-1], num_groups=norm_groups),
            AttentionBlock(widths[-1], num_groups=norm_groups),
            ResidualBlock(widths[-1], widths[-1], num_groups=norm_groups)
        ])

        for i in reversed(range(len(widths))):
            in_channels = widths[i] * 2 if i != len(widths) - 1 else widths[i]
            out_channels = widths[i-1] if i != 0 else base_channels
            self.ups.append(nn.ModuleList([
                ResidualBlock(in_channels, out_channels, num_groups=norm_groups),
                ResidualBlock(out_channels, out_channels, num_groups=norm_groups),
                UpSample(out_channels) if i != 0 else nn.Identity()
            ]))
            if has_attention[i]:
                self.ups[-1].append(AttentionBlock(out_channels, num_groups=norm_groups))

        self.out_conv = nn.Sequential(
            nn.GroupNorm(num_groups=norm_groups, num_channels=base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, condition):
        x = torch.cat([x, condition], dim=1)
        t = self.time_embed(t)
        x = self.in_conv(x)
        skips = []
        for down in self.downs:
            for layer in down:
                x = layer(x) if not isinstance(layer, AttentionBlock) else layer(x)
            skips.append(x)
        for layer in self.middle:
            x = layer(x)
        for i, up in enumerate(self.ups):
            skip = skips.pop()
            if x.shape[2:] != skip.shape[2:]:
                skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1) if i != 0 else x
            for layer in up:
                x = layer(x) if not isinstance(layer, AttentionBlock) else layer(x)
        return self.out_conv(x)

class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Initialize the model and EMA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet(img_channels, first_conv_channels, widths, has_attention).to(device)
ema_unet = UNet(img_channels, first_conv_channels, widths, has_attention).to(device)
ema_unet.load_state_dict(unet.state_dict())
ema = EMA(ema_unet, 0.999)
ema.register()

# Optimizer with learning rate scheduler
optimizer = optim.Adam(unet.parameters(), lr=initial_learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

criterion = nn.MSELoss()

# Training loop
diffusion = GaussianDiffusion(timesteps=total_timesteps)

def train(epoch):
    unet.train()
    for batch_idx, batch in enumerate(train_loader):
        images = batch[0].to(device)
        batch_size = images.size(0)
        t = torch.randint(0, total_timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(images)
        x_t = diffusion.q_sample(images, t, noise)

        optimizer.zero_grad()
        pred_noise = unet(x_t, t, images)  # Condition on input images
        loss = criterion(pred_noise, noise)
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)

        optimizer.step()
        ema.update()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1}/{num_epochs} completed.")
    scheduler.step()  # Step the learning rate scheduler

    # Save the model checkpoints
    torch.save(unet.state_dict(), os.path.join(checkpoint_dir, f"unet_epoch_{epoch+1}.pth"))
    torch.save(ema_unet.state_dict(), os.path.join(checkpoint_dir, f"ema_unet_epoch_{epoch+1}.pth"))

def generate_samples(condition_images):
    unet.eval()
    ema.apply_shadow()
    with torch.no_grad():
        samples = torch.randn(10, img_channels, img_size, img_size, device=device)
        for i in reversed(range(total_timesteps)):
            t = torch.full((10,), i, device=device, dtype=torch.long)
            pred_noise = ema_unet(samples, t, condition_images[:10])
            posterior_mean, _, _ = diffusion.q_posterior(pred_noise, samples, t)
            samples = posterior_mean + torch.randn_like(samples) * torch.sqrt(diffusion.posterior_variance[i])
    
    ema.restore()
    
    samples = (samples * 0.5 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(samples, nrow=4)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.show()

# To load the model checkpoint later for inference or to continue training
def load_checkpoint(epoch):
    unet.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"unet_epoch_{epoch}.pth")))
    ema_unet.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"ema_unet_epoch_{epoch}.pth")))
    print(f"Loaded model checkpoints from epoch {epoch}")

# Training loop without generating samples at each epoch
for epoch in range(num_epochs):
    train(epoch)
    
# Generate samples after the entire training is complete
condition_images = next(iter(train_loader))[0].to(device)
generate_samples(condition_images)

# Example: load_checkpoint(10) to load the model from epoch 10
