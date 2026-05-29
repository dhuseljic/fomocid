"""Shared SSL model, optimizer, and scheduler utilities."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models

from fomocid.config_types import OptimizerConfig, RootConfig, SchedulerConfig


def build_resnet_encoder(backbone_name: str, image_size: int) -> tuple[nn.Module, int]:
    """Build a ResNet encoder adapted for tutorial image sizes.

    Args:
        backbone_name: Symbolic backbone name.
        image_size: Input image size used by the tutorial.

    Returns:
        A tuple of ``(encoder, feature_dim)``.

    Raises:
        ValueError: If the backbone name is unsupported.

    Parameters
    ----------
    backbone_name : str
        Input value for ``backbone_name``.
    image_size : int
        Input value for ``image_size``.

    Returns
    -------
    result : tuple[nn.Module, int]
        Return value produced by the function.
    """
    if backbone_name != "resnet18":
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    backbone = models.resnet18(weights=None)
    if image_size <= 64:
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim


def freeze_module(module: nn.Module) -> None:
    """Put a module in eval mode and disable gradients for all parameters.

    Args:
        module: Module to freeze.

    Parameters
    ----------
    module : nn.Module
        Input value for ``module``.

    Returns
    -------
    None
        The function completes in place.
    """
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


@torch.no_grad()
def momentum_update(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """Update a teacher network with exponential moving averages.

    Args:
        student: Online network providing current weights.
        teacher: Momentum network updated in place.
        momentum: Exponential moving-average coefficient.

    Parameters
    ----------
    student : nn.Module
        Input value for ``student``.
    teacher : nn.Module
        Input value for ``teacher``.
    momentum : float
        Input value for ``momentum``.

    Returns
    -------
    None
        The function completes in place.
    """
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters(), strict=True):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)

    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        if name in student_buffers:
            teacher_buffer.copy_(student_buffers[name])


def resolve_optimizer_config(config: RootConfig) -> OptimizerConfig:
    """Validate and normalize optimizer settings from a config.

    Args:
        config: Normalized root config. Only the ``optimizer`` section is
            consumed here, with ``lr``, ``weight_decay``, and the optional
            ``optimizer.scheduler`` subsection.

    Returns:
        A normalized optimizer configuration dictionary.

    Raises:
        ValueError: If optimizer parameters are invalid.

    Parameters
    ----------
    config : RootConfig
        Input value for ``config``.

    Returns
    -------
    result : OptimizerConfig
        Return value produced by the function.
    """
    optimizer_config = dict(config.get("optimizer", {}))
    learning_rate = float(optimizer_config.get("lr", 1e-3))
    weight_decay = float(optimizer_config.get("weight_decay", 1e-4))
    if learning_rate <= 0.0:
        raise ValueError("optimizer.lr must be greater than 0.")
    if weight_decay < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative.")

    scheduler_config = _resolve_scheduler_config(optimizer_config.get("scheduler"), learning_rate)
    return {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": scheduler_config,
    }


def _resolve_scheduler_config(raw_scheduler_config: Any, learning_rate: float) -> SchedulerConfig | None:
    """Validate optional scheduler settings for SSL pretraining.

    Args:
        raw_scheduler_config: Unnormalized scheduler subsection from the
            optimizer config.
        learning_rate: Resolved peak optimizer learning rate used to validate
            the scheduler floor.

    Returns:
        Normalized scheduler config, or ``None`` when scheduling is disabled.

    Raises:
        ValueError: If the scheduler name, interval, or numeric bounds are
            invalid.

    Parameters
    ----------
    raw_scheduler_config : Any
        Input value for ``raw_scheduler_config``.
    learning_rate : float
        Input value for ``learning_rate``.

    Returns
    -------
    result : SchedulerConfig | None
        Return value produced by the function.
    """
    if raw_scheduler_config is None:
        return None

    scheduler_config = dict(raw_scheduler_config)
    if not bool(scheduler_config.get("enabled", False)):
        return None

    name = str(scheduler_config.get("name", "cosine_with_warmup"))
    if name != "cosine_with_warmup":
        raise ValueError(f"Unsupported scheduler: {name}. Supported scheduler: cosine_with_warmup")

    interval = str(scheduler_config.get("interval", "step"))
    if interval != "step":
        raise ValueError("optimizer.scheduler.interval must be 'step' in phase 1.")

    warmup_epochs = float(scheduler_config.get("warmup_epochs", 0.0))
    min_lr = float(scheduler_config.get("min_lr", 0.0))
    if warmup_epochs < 0.0:
        raise ValueError("optimizer.scheduler.warmup_epochs must be non-negative.")
    if min_lr < 0.0:
        raise ValueError("optimizer.scheduler.min_lr must be non-negative.")
    if min_lr > learning_rate:
        raise ValueError("optimizer.scheduler.min_lr must be less than or equal to optimizer.lr.")

    return {
        "name": name,
        "enabled": True,
        "warmup_epochs": warmup_epochs,
        "min_lr": min_lr,
        "interval": interval,
    }


def build_optimizer(parameters: Any, optimizer_config: OptimizerConfig) -> Optimizer:
    """Create the AdamW optimizer used for SSL pretraining.

    Args:
        parameters: Iterable of trainable parameters.
        optimizer_config: Normalized optimizer config containing ``lr`` and
            ``weight_decay``.

    Returns:
        Configured AdamW optimizer.

    Parameters
    ----------
    parameters : Any
        Input value for ``parameters``.
    optimizer_config : OptimizerConfig
        Input value for ``optimizer_config``.

    Returns
    -------
    result : Optimizer
        Return value produced by the function.
    """
    return AdamW(parameters, lr=float(optimizer_config["lr"]), weight_decay=float(optimizer_config["weight_decay"]))


def estimate_warmup_steps(total_steps: int, max_epochs: int, warmup_epochs: float) -> int:
    """Convert epoch-based warmup into an integer number of optimizer steps.

    Args:
        total_steps: Estimated number of optimizer steps for the full run.
        max_epochs: Number of training epochs.
        warmup_epochs: Warmup duration expressed in epochs.

    Returns:
        Number of optimizer steps that should use linear warmup.

    Parameters
    ----------
    total_steps : int
        Input value for ``total_steps``.
    max_epochs : int
        Input value for ``max_epochs``.
    warmup_epochs : float
        Input value for ``warmup_epochs``.

    Returns
    -------
    result : int
        Return value produced by the function.
    """
    if warmup_epochs <= 0.0:
        return 0
    bounded_total_steps = max(int(total_steps), 1)
    bounded_max_epochs = max(int(max_epochs), 1)
    steps_per_epoch = max(float(bounded_total_steps) / float(bounded_max_epochs), 1.0)
    return max(int(math.ceil(warmup_epochs * steps_per_epoch)), 0)


def warmup_cosine_factor(
    step_index: int,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> float:
    """Compute the multiplicative learning-rate factor for one step.

    Args:
        step_index: Zero-based optimizer step index.
        total_steps: Total number of scheduled steps in the run.
        warmup_steps: Number of steps reserved for linear warmup.
        min_lr_scale: Ratio between the minimum LR and peak LR.

    Returns:
        Multiplicative factor applied by :class:`torch.optim.lr_scheduler.LambdaLR`.

    Parameters
    ----------
    step_index : int
        Input value for ``step_index``.
    total_steps : int
        Input value for ``total_steps``.
    warmup_steps : int
        Input value for ``warmup_steps``.
    min_lr_scale : float
        Input value for ``min_lr_scale``.

    Returns
    -------
    result : float
        Return value produced by the function.
    """
    bounded_total_steps = max(int(total_steps), 1)
    bounded_step_index = max(int(step_index), 0)
    bounded_min_lr_scale = min(max(float(min_lr_scale), 0.0), 1.0)

    if warmup_steps > 0 and bounded_total_steps <= warmup_steps:
        return min(float(bounded_step_index + 1) / float(bounded_total_steps), 1.0)

    if warmup_steps > 0 and bounded_step_index < warmup_steps:
        return min(float(bounded_step_index + 1) / float(warmup_steps), 1.0)

    remaining_steps = max(bounded_total_steps - warmup_steps, 1)
    if remaining_steps == 1:
        return bounded_min_lr_scale

    progress = min(float(bounded_step_index - warmup_steps) / float(remaining_steps - 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return bounded_min_lr_scale + (1.0 - bounded_min_lr_scale) * cosine


def build_scheduler(
    optimizer: Optimizer,
    optimizer_config: OptimizerConfig,
    *,
    total_steps: int,
    max_epochs: int,
) -> LambdaLR | None:
    """Create the optional warmup-plus-cosine scheduler for pretraining.

    Args:
        optimizer: Optimizer to schedule.
        optimizer_config: Normalized optimizer configuration.
        total_steps: Total estimated optimizer steps.
        max_epochs: Number of training epochs.

    Returns:
        A configured scheduler, or ``None`` if scheduling is disabled.

    Parameters
    ----------
    optimizer : Optimizer
        Input value for ``optimizer``.
    optimizer_config : OptimizerConfig
        Input value for ``optimizer_config``.
    total_steps : int
        Input value for ``total_steps``.
    max_epochs : int
        Input value for ``max_epochs``.

    Returns
    -------
    result : LambdaLR | None
        Return value produced by the function.
    """
    scheduler_config = optimizer_config.get("scheduler")
    if scheduler_config is None:
        return None

    warmup_steps = estimate_warmup_steps(
        total_steps=total_steps,
        max_epochs=max_epochs,
        warmup_epochs=float(scheduler_config["warmup_epochs"]),
    )
    min_lr_scale = float(scheduler_config["min_lr"]) / float(optimizer_config["lr"])
    return LambdaLR(
        optimizer,
        lr_lambda=lambda step_index: warmup_cosine_factor(
            step_index,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            min_lr_scale=min_lr_scale,
        ),
    )


def configure_pretraining_optimizers(module: Any, optimizer_config: OptimizerConfig) -> Any:
    """Build the Lightning optimizer configuration for SSL modules.

    Args:
        module: Lightning module being optimized.
        optimizer_config: Normalized optimizer config produced by
            :func:`resolve_optimizer_config`.

    Returns:
        Either an optimizer or a Lightning optimizer-and-scheduler mapping.

    Parameters
    ----------
    module : Any
        Input value for ``module``.
    optimizer_config : OptimizerConfig
        Input value for ``optimizer_config``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    optimizer = build_optimizer(module.parameters(), optimizer_config)
    scheduler = build_scheduler(
        optimizer,
        optimizer_config,
        total_steps=max(int(module.trainer.estimated_stepping_batches), 1),
        max_epochs=max(int(module.trainer.max_epochs), 1),
    )
    if scheduler is None:
        return optimizer

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        },
    }
