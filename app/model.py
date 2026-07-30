"""
Siamese Facial Recognition - Embedding network (shared twin branch).

Architecture: a ResNet-18-style CNN built FROM SCRATCH (no pretrained weights),
augmented with Squeeze-and-Excitation (SE) channel attention for higher accuracy
under few-shot conditions. Outputs an L2-normalized 512-d embedding.

IMPORTANT: This file must stay byte-identical (in behaviour) to the
`EmbeddingNet` definition inside `kaggle/train_siamese.ipynb`. The trained
`.pth` checkpoint is loaded directly by the inference / Gradio app.

References:
  - He et al., "Deep Residual Learning for Image Recognition" (2015)  -> ResNet
  - Hu et al., "Squeeze-and-Excitation Networks" (2017)              -> SE block
  - Schroff et al., "FaceNet" (2015)                                 -> embedding + triplet
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation channel attention block
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Re-calibrates channel-wise feature responses. Cheap and helps few-shot."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ---------------------------------------------------------------------------
# Basic residual block (ResNet "BasicBlock") + SE
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.se = SEBlock(out_ch, reduction=reduction)

        # Downsample identity path when spatial dims / channels change
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out


# ---------------------------------------------------------------------------
# Full embedding network (the shared twin branch of the Siamese network)
# ---------------------------------------------------------------------------
class EmbeddingNet(nn.Module):
    """
    ResNet-18-style CNN (from scratch) producing a 512-d embedding.

    Input : NCHW float tensor, shape (B, 3, 112, 112), normalized to [0,1].
    Output: (B, embedding_dim) L2-normalized embedding.
    """

    def __init__(self, embedding_dim: int = 512, dropout: float = 0.4):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stem (keeps spatial size, then halves twice via first block strides).
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage blocks: [64, 128, 256, 512], each spatial /2 (except stage 1).
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

        self._init_weights()

    @staticmethod
    def _make_layer(in_ch: int, out_ch: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        # L2-normalize so cosine similarity == dot product, triplet margin well-scaled.
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Siamese wrapper (two shared-weight branches) - used conceptually / for verify
# ---------------------------------------------------------------------------
class SiameseNet(nn.Module):
    """Wraps EmbeddingNet into a two-branch Siamese form for verification."""

    def __init__(self, embedding_net: EmbeddingNet):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        # Euclidean distance between normalized embeddings (== sqrt(2-2*cos)).
        dist = F.pairwise_distance(e1, e2, p=2)
        return e1, e2, dist


# ---------------------------------------------------------------------------
# Convenience loader used by the inference / Gradio app
# ---------------------------------------------------------------------------
def load_embedding_net(checkpoint_path: str,
                       map_location: str = "cpu",
                       embedding_dim: int = 512) -> EmbeddingNet:
    """
    Reconstruct EmbeddingNet and load trained weights.

    Handles three checkpoint layouts:
      - raw state_dict
      - {"state_dict": ...}  (EMA / best checkpoint)
      - {"model_state_dict": ...}
    """
    net = EmbeddingNet(embedding_dim=embedding_dim)
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "ema_state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    net.load_state_dict(ckpt)
    net.eval()
    return net


if __name__ == "__main__":
    # Quick self-test: verify forward pass + output shape/norm.
    net = EmbeddingNet()
    x = torch.rand(2, 3, 112, 112)
    e = net(x)
    norms = e.norm(p=2, dim=1)
    print(f"Embedding shape: {tuple(e.shape)}")
    print(f"L2 norms (should be ~1.0): {norms.tolist()}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {n_params:,} (~{n_params/1e6:.2f}M)")
