"""MAE Lightning module used by the CIFAR-10 tutorial workflow."""

from __future__ import annotations

from typing import Any, cast

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightly.models import utils as lightly_utils
from lightly.models.modules.masked_autoencoder import MAEBackbone, MAEDecoder
from torch import nn
from torchvision.models import VisionTransformer

from fomocid.config_types import MAEConfig, RootConfig

from .common import configure_pretraining_optimizers, resolve_optimizer_config


class MAEModule(pl.LightningModule):
    """Masked Autoencoder Lightning module used in the CIFAR-10 tutorial."""

    def __init__(self, config: RootConfig) -> None:
        """Build the MAE encoder, decoder, and optimization setup.

        Args:
            config: Normalized root config. This module reads
                ``dataset.image_size`` plus MAE-specific ``ssl`` keys such as
                ``patch_size``, ``mask_ratio``, encoder/decoder dimensions, and
                ``normalize_pixel_targets``.

        Raises:
            ValueError: If the image size is incompatible with the patch size.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config})

        dataset_config = config["dataset"]
        ssl_config = cast(MAEConfig, config["ssl"])

        self.image_size = int(dataset_config["image_size"])
        self.patch_size = int(ssl_config.get("patch_size", 4))
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size for MAE.")

        hidden_dim = int(ssl_config.get("encoder_hidden_dim", 256))
        vit = VisionTransformer(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_layers=int(ssl_config.get("encoder_depth", 6)),
            num_heads=int(ssl_config.get("encoder_num_heads", 8)),
            hidden_dim=hidden_dim,
            mlp_dim=int(ssl_config.get("encoder_mlp_dim", hidden_dim * 4)),
            dropout=float(ssl_config.get("dropout", 0.0)),
            attention_dropout=float(ssl_config.get("attention_dropout", 0.0)),
        )
        self.backbone = MAEBackbone.from_vit(vit)
        self.decoder = MAEDecoder(
            seq_length=self.backbone.seq_length,
            num_layers=int(ssl_config.get("decoder_depth", 2)),
            num_heads=int(ssl_config.get("decoder_num_heads", 8)),
            embed_input_dim=hidden_dim,
            hidden_dim=int(ssl_config.get("decoder_hidden_dim", 128)),
            mlp_dim=int(ssl_config.get("decoder_mlp_dim", 256)),
            out_dim=3 * self.patch_size * self.patch_size,
            dropout=float(ssl_config.get("dropout", 0.0)),
            attention_dropout=float(ssl_config.get("attention_dropout", 0.0)),
        )
        self.decoder_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.decoder_mask_token, std=0.02)

        self.feature_dim = hidden_dim
        self.mask_ratio = float(ssl_config.get("mask_ratio", 0.75))
        self.normalize_pixel_targets = bool(ssl_config.get("normalize_pixel_targets", True))
        self.optimizer_config = resolve_optimizer_config(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Alias :meth:`embed` so the module can be called like a feature encoder.

        Args:
            inputs: Image batch with shape ``(batch, channels, height, width)``.

        Returns:
            Image-level embeddings derived from MAE encoder tokens.
        """
        return self.embed(inputs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode images into a single MAE feature vector per sample.

        Args:
            inputs: Image batch with shape ``(batch, channels, height, width)``.

        Returns:
            Feature matrix with one pooled embedding per image.
        """
        encoded = self.backbone.encode(inputs)
        return self._pool_tokens(encoded)

    def training_step(self, batch: Any, _batch_idx: int) -> torch.Tensor:
        """Run one MAE pretraining step and log masked-patch reconstruction loss.

        Args:
            batch: Lightning batch containing MAE views and labels.
            _batch_idx: Zero-based batch index inside the current epoch.

        Returns:
            Scalar masked-reconstruction loss tensor for the current step.
        """
        views, _targets = batch
        images = views[0]
        idx_keep, idx_mask = lightly_utils.random_token_mask(
            size=(images.size(0), self.backbone.seq_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        encoded = self.backbone.encode(images, idx_keep=idx_keep)
        decoder_input = self.decoder_mask_token.repeat(images.size(0), self.backbone.seq_length, 1)
        decoder_input = lightly_utils.set_at_index(decoder_input, idx_keep, encoded)
        predictions = self.decoder(decoder_input)

        patch_targets = lightly_utils.patchify(images, self.patch_size)
        if self.normalize_pixel_targets:
            mean = patch_targets.mean(dim=-1, keepdim=True)
            var = patch_targets.var(dim=-1, keepdim=True, unbiased=False)
            patch_targets = (patch_targets - mean) / (var + 1e-6).sqrt()

        masked_predictions = lightly_utils.get_at_index(predictions, idx_mask)
        masked_targets = lightly_utils.get_at_index(patch_targets, idx_mask - 1)
        loss = F.mse_loss(masked_predictions, masked_targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def configure_optimizers(self) -> Any:
        """Build the AdamW optimizer and optional LR scheduler."""
        return configure_pretraining_optimizers(self, self.optimizer_config)

    def _pool_tokens(self, encoded_tokens: torch.Tensor) -> torch.Tensor:
        """Reduce MAE token sequences to image-level embeddings for evaluation."""
        if encoded_tokens.ndim != 3:
            return encoded_tokens

        pooling = str(self.config["ssl"].get("embedding_pool", "cls")).lower()
        if pooling == "mean":
            # Exclude the class token so the representation summarizes patches.
            return encoded_tokens[:, 1:, :].mean(dim=1)
        if pooling != "cls":
            raise ValueError(f"Unsupported ssl.embedding_pool value: {pooling}. Supported values: cls, mean")
        return encoded_tokens[:, 0, :]
