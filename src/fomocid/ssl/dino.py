"""DINO Lightning module used by the STL-10 tutorial workflow."""

from __future__ import annotations

from typing import Any, cast

import pytorch_lightning as pl
import torch
from lightly.loss import DINOLoss
from lightly.models.modules.heads import DINOProjectionHead

from fomocid.config_types import DINOConfig, RootConfig

from .common import (
    build_resnet_encoder,
    configure_pretraining_optimizers,
    freeze_module,
    momentum_update,
    resolve_optimizer_config,
)


class DINOModule(pl.LightningModule):
    """DINO Lightning module used in the STL-10 tutorial."""

    def __init__(self, config: RootConfig) -> None:
        """Build the DINO student, teacher, heads, and loss.

        Args:
            config: Normalized root config. This module consumes
                ``dataset.image_size`` plus DINO-specific ``ssl`` keys such as
                ``backbone``, crop parameters, projection-head sizes, teacher
                temperatures, and EMA momentum bounds.

        Parameters
        ----------
        config : RootConfig
            Input value for ``config``.

        Returns
        -------
        None
            The function completes in place.
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters({"config": config})

        dataset_config = config["dataset"]
        ssl_config = cast(DINOConfig, config["ssl"])

        image_size = int(dataset_config["image_size"])
        backbone_name = ssl_config.get("backbone", "resnet18")
        self.student_backbone, feature_dim = build_resnet_encoder(backbone_name, image_size=image_size)
        self.teacher_backbone, _ = build_resnet_encoder(backbone_name, image_size=image_size)
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())

        head_kwargs = {
            "input_dim": feature_dim,
            "hidden_dim": int(ssl_config.get("projection_hidden_dim", 1024)),
            "bottleneck_dim": int(ssl_config.get("projection_bottleneck_dim", 256)),
            "output_dim": int(ssl_config.get("projection_output_dim", 4096)),
            "batch_norm": bool(ssl_config.get("projection_batch_norm", True)),
            "freeze_last_layer": int(ssl_config.get("freeze_last_layer_epochs", 1)),
            "norm_last_layer": bool(ssl_config.get("norm_last_layer", True)),
        }
        self.student_head = DINOProjectionHead(**head_kwargs)
        self.teacher_head = DINOProjectionHead(**head_kwargs)
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        freeze_module(self.teacher_backbone)
        freeze_module(self.teacher_head)

        self.feature_dim = feature_dim
        self.loss_fn = DINOLoss(
            output_dim=head_kwargs["output_dim"],
            warmup_teacher_temp=float(ssl_config.get("warmup_teacher_temp", 0.04)),
            teacher_temp=float(ssl_config.get("teacher_temp", 0.04)),
            warmup_teacher_temp_epochs=int(ssl_config.get("warmup_teacher_temp_epochs", 10)),
            student_temp=float(ssl_config.get("student_temp", 0.1)),
            center_momentum=float(ssl_config.get("center_momentum", 0.9)),
        )
        self.teacher_momentum_start = float(ssl_config.get("teacher_momentum_start", 0.996))
        self.teacher_momentum_end = float(ssl_config.get("teacher_momentum_end", 1.0))
        self.optimizer_config = resolve_optimizer_config(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Alias :meth:`embed` so the module can act as a feature extractor.

        Args:
            inputs: Image batch with shape ``(batch, channels, height, width)``.

        Returns:
            Image-level embeddings from the student backbone.

        Parameters
        ----------
        inputs : torch.Tensor
            Input value for ``inputs``.

        Returns
        -------
        result : torch.Tensor
            Return value produced by the function.
        """
        return self.embed(inputs)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode images with the student backbone.

        Args:
            inputs: Image batch with shape ``(batch, channels, height, width)``.

        Returns:
            Feature matrix with one embedding vector per image.

        Parameters
        ----------
        inputs : torch.Tensor
            Input value for ``inputs``.

        Returns
        -------
        result : torch.Tensor
            Return value produced by the function.
        """
        return self.student_backbone(inputs)

    def train(self, mode: bool = True) -> "DINOModule":
        """Keep the teacher networks in eval mode during training.

        Args:
            mode: Requested training mode for the student branch.

        Returns:
            The module instance for chaining.

        Parameters
        ----------
        mode : bool
            Input value for ``mode``.

        Returns
        -------
        result : "DINOModule"
            Return value produced by the function.
        """
        super().train(mode)
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        return self

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Run one DINO student-teacher optimization step.

        Args:
            batch: Lightning batch containing the multi-crop views and labels.
            batch_idx: Zero-based batch index inside the current epoch.

        Returns:
            Scalar DINO loss tensor for the current optimization step.

        Parameters
        ----------
        batch : Any
            Input value for ``batch``.
        batch_idx : int
            Input value for ``batch_idx``.

        Returns
        -------
        result : torch.Tensor
            Return value produced by the function.
        """
        views, _targets = batch
        student_outputs = [self.student_head(self.student_backbone(view)) for view in views]
        with torch.no_grad():
            teacher_outputs = [self.teacher_head(self.teacher_backbone(view)) for view in views[:2]]
        loss = self.loss_fn(teacher_outputs, student_outputs, epoch=self.current_epoch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=views[0].size(0))
        return loss

    def on_after_backward(self) -> None:
        """Apply DINO's last-layer gradient cancellation schedule.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        None
            The function completes in place.
        """
        self.student_head.cancel_last_layer_gradients(self.current_epoch)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Update teacher parameters with momentum after each batch.

        Args:
            outputs: Output returned by :meth:`training_step`.
            batch: Current training batch.
            batch_idx: Zero-based batch index inside the current epoch.

        Parameters
        ----------
        outputs : Any
            Input value for ``outputs``.
        batch : Any
            Input value for ``batch``.
        batch_idx : int
            Input value for ``batch_idx``.

        Returns
        -------
        None
            The function completes in place.
        """
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        progress = min(float(self.global_step) / float(total_steps), 1.0)
        momentum = self.teacher_momentum_start + (self.teacher_momentum_end - self.teacher_momentum_start) * progress
        momentum_update(self.student_backbone, self.teacher_backbone, momentum=momentum)
        momentum_update(self.student_head, self.teacher_head, momentum=momentum)
        views, _targets = batch
        self.log("teacher_momentum", momentum, on_step=True, on_epoch=False, prog_bar=False, batch_size=views[0].size(0))

    def configure_optimizers(self) -> Any:
        """Build the AdamW optimizer and optional LR scheduler.

        Parameters
        ----------
        None
            This function takes no explicit input parameters.

        Returns
        -------
        result : Any
            Return value produced by the function.
        """
        return configure_pretraining_optimizers(self, self.optimizer_config)
