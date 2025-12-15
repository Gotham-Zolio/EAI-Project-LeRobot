import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        vision_backbone: str = "resnet18",
        num_diffusion_steps: int = 100,
        down_dims: Tuple[int, ...] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.num_diffusion_steps = num_diffusion_steps

        # Vision Encoder
        # We assume one camera for simplicity, but can be extended
        if vision_backbone == "resnet18":
            self.vision_encoder = torchvision.models.resnet18(pretrained=True)
            self.vision_encoder.fc = nn.Identity()  # Remove classification head
            self.vision_feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {vision_backbone}")

        # State Encoder (MLP)
        self.state_encoder = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.state_feature_dim = 64

        # Global Condition Dimension
        self.global_cond_dim = self.vision_feature_dim + self.state_feature_dim

        # Noise Prediction Network (Conditional MLP)
        self.noise_pred_net = ConditionalMLP(
            input_dim=action_dim,
            global_cond_dim=self.global_cond_dim,
            hidden_dim=256,
            num_layers=4
        )

        # Noise Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_steps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Training forward pass.
        Returns loss.
        """
        # 1. Encode observations
        obs_state = batch["observation.state"]
        obs_image = batch["observation.images.front"]  # (B, C, H, W)

        state_features = self.state_encoder(obs_state)
        vision_features = self.vision_encoder(obs_image)
        
        # Concatenate features
        global_cond = torch.cat([vision_features, state_features], dim=-1)

        # 2. Sample noise
        actions = batch["action"]
        noise = torch.randn_like(actions)
        bsz = actions.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=actions.device
        ).long()

        # 3. Add noise to the clean action according to the noise magnitude at each timestep
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        # 4. Predict the noise residual
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, global_cond=global_cond)

        # 5. Calculate loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference forward pass.
        Returns predicted action.
        """
        obs_state = batch["observation.state"]
        obs_image = batch["observation.images.front"]
        
        bsz = obs_state.shape[0]

        # 1. Encode observations
        state_features = self.state_encoder(obs_state)
        vision_features = self.vision_encoder(obs_image)
        global_cond = torch.cat([vision_features, state_features], dim=-1)

        # 2. Initialize noisy action
        noisy_action = torch.randn(
            (bsz, self.action_dim), device=obs_state.device
        )

        # 3. Denoising loop
        self.noise_scheduler.set_timesteps(self.num_diffusion_steps)

        for t in self.noise_scheduler.timesteps:
            # Ensure t is on the same device as the model
            t = t.to(obs_state.device)
            
            # Predict noise
            model_output = self.noise_pred_net(
                noisy_action, t, global_cond=global_cond
            )

            # Compute previous noisy sample x_t -> x_t-1
            noisy_action = self.noise_scheduler.step(
                model_output, t, noisy_action
            ).prev_sample

        return noisy_action


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        ds = []
        for i in range(len(down_dims)):
            in_dim = all_dims[i]
            out_dim = all_dims[i + 1]
            ds.append(
                ConditionalResidualBlock1D(
                    in_dim, out_dim, global_cond_dim, kernel_size, n_groups
                )
            )
            if i < len(down_dims) - 1:
                ds.append(nn.Conv1d(out_dim, out_dim, 3, 2, 1)) # Downsample

        self.down_modules = nn.ModuleList(ds)

        us = []
        for i in range(len(down_dims) - 1, -1, -1):
            in_dim = all_dims[i + 1]
            out_dim = all_dims[i]
            
            # Skip connection adds input_dim
            # But here we are doing a simple structure without U-Net skip connections for simplicity first?
            # Wait, U-Net needs skip connections.
            # Let's implement a simpler MLP-based diffusion or a proper U-Net.
            # Given the complexity, I'll implement a Residual MLP for 1D data which is common for state-based,
            # but for actions, 1D CNN is good.
            # Let's stick to a simple Residual MLP structure if dimensions are small (action dim ~6).
            # Actually, for action chunking, 1D CNN is used. But here we predict single action step?
            # The dataset returns single action. So input is (B, ActionDim).
            # 1D CNN is for (B, T, ActionDim).
            # If we only predict single step, MLP is enough.
            pass
        
        # Re-thinking: Single step action prediction.
        # Input: (B, ActionDim).
        # We can treat it as (B, ActionDim, 1) or just use MLP.
        # Let's use a Residual MLP (ResNet-style) which is very effective for vector inputs.
        
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], global_cond_dim, kernel_size, n_groups),
            ConditionalResidualBlock1D(down_dims[-1], down_dims[-1], global_cond_dim, kernel_size, n_groups),
        ])

        self.up_modules = nn.ModuleList()
        for i in range(len(down_dims) - 1, -1, -1):
            in_dim = all_dims[i + 1]
            out_dim = all_dims[i]
            # No skip connections from encoder in this simplified version, just upsampling
            # To make it a real U-Net we need to store encoder outputs.
            # For simplicity and robustness in this demo, I will use a symmetric structure but without long skip connections
            # or just a simple MLP structure.
            # Let's go with a robust MLP structure for single-step action.
            pass


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_dim, out_dim, global_cond_dim, kernel_size=5, n_groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_dim),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_dim),
            nn.Mish(),
        )
        self.cond_proj = nn.Linear(global_cond_dim, out_dim * 2)
        self.residual = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x, global_cond):
        # x: (B, C, T)
        # global_cond: (B, GlobalCondDim)
        
        h = self.block1(x)
        
        # FiLM conditioning
        cond = self.cond_proj(global_cond) # (B, 2*C)
        cond = cond.unsqueeze(-1) # (B, 2*C, 1)
        scale, shift = torch.chunk(cond, 2, dim=1)
        
        h = h * (1 + scale) + shift
        h = self.block2(h)
        
        return h + self.residual(x)


class ConditionalMLP(nn.Module):
    def __init__(self, input_dim, global_cond_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cond_proj = nn.Linear(global_cond_dim, hidden_dim)
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.layers = nn.ModuleList([
            ResidualMLPBlock(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, time, global_cond):
        # x: (B, ActionDim)
        # time: (B,)
        # global_cond: (B, GlobalCondDim)
        
        t_emb = self.time_emb(time)
        cond_emb = self.cond_proj(global_cond)
        x_emb = self.input_proj(x)
        
        h = x_emb + t_emb + cond_emb
        
        for layer in self.layers:
            h = layer(h)
            
        return self.out_proj(h)


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish(),
        )
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.residual(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)
            
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
