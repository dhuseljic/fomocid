"""Typed configuration interfaces and normalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypedDict, cast


DatasetName = Literal["cifar10", "mnist", "stl10"]
ProjectionMethod = Literal["pca", "tsne"]
ProgressBarStyle = Literal["auto", "rich", "tqdm"]
EmbeddingPool = Literal["cls", "mean"]
SchedulerName = Literal["cosine_with_warmup"]

_SUPPORTED_DATASETS: dict[DatasetName, dict[str, Any]] = {
    "cifar10": {
        "image_size": 32,
        "pretrain_split": "train",
        "train_split": "train",
        "test_split": "test",
    },
    "mnist": {
        "image_size": 28,
        "pretrain_split": "train",
        "train_split": "train",
        "test_split": "test",
    },
    "stl10": {
        "image_size": 96,
        "pretrain_split": "train+unlabeled",
        "train_split": "train",
        "test_split": "test",
    },
}


class DatasetConfig(TypedDict):
    """Normalized dataset configuration for benchmark tutorials."""

    name: DatasetName
    data_dir: str
    image_size: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    download: bool
    use_fake_data: bool
    fake_data_size: int
    pretrain_split: str
    train_split: str
    test_split: str


class MAEConfig(TypedDict):
    """Normalized SSL configuration for MAE tutorials."""

    method: Literal["mae"]
    patch_size: int
    mask_ratio: float
    min_scale: float
    encoder_depth: int
    encoder_num_heads: int
    encoder_hidden_dim: int
    encoder_mlp_dim: int
    decoder_depth: int
    decoder_num_heads: int
    decoder_hidden_dim: int
    decoder_mlp_dim: int
    normalize_pixel_targets: bool
    embedding_pool: EmbeddingPool
    dropout: float
    attention_dropout: float


class DINOConfig(TypedDict):
    """Normalized SSL configuration for DINO tutorials."""

    method: Literal["dino"]
    backbone: str
    global_crop_size: int
    global_crop_scale: list[float]
    local_crop_size: int
    local_crop_scale: list[float]
    n_local_views: int
    projection_hidden_dim: int
    projection_bottleneck_dim: int
    projection_output_dim: int
    projection_batch_norm: bool
    freeze_last_layer_epochs: int
    norm_last_layer: bool
    teacher_momentum_start: float
    teacher_momentum_end: float
    warmup_teacher_temp: float
    teacher_temp: float
    warmup_teacher_temp_epochs: int
    student_temp: float
    center_momentum: float
    color_jitter_strength: float
    random_gray_scale: float
    gaussian_blur: list[float]
    solarization_prob: float


SSLConfig = MAEConfig | DINOConfig


class SchedulerConfig(TypedDict):
    """Normalized scheduler configuration for SSL optimization."""

    enabled: bool
    name: SchedulerName
    warmup_epochs: float
    min_lr: float
    interval: Literal["step"]


class OptimizerConfig(TypedDict):
    """Normalized optimizer configuration."""

    lr: float
    weight_decay: float
    scheduler: SchedulerConfig | None


class TrainerConfig(TypedDict):
    """Normalized Lightning trainer configuration."""

    accelerator: str | int | list[int]
    devices: str | int | list[int]
    max_epochs: int
    limit_train_batches: float | int
    log_every_n_steps: int
    enable_logging: bool
    enable_progress_bar: bool
    progress_bar: ProgressBarStyle
    progress_refresh_rate: int
    deterministic: bool
    fast_dev_run: bool


class LinearProbeConfig(TypedDict):
    """Normalized linear-probe evaluation configuration."""

    epochs: int
    lr: float
    weight_decay: float
    batch_size: int


class KNNConfig(TypedDict):
    """Normalized k-nearest-neighbor evaluation configuration."""

    k: int
    chunk_size: int


class EvaluationConfig(TypedDict):
    """Normalized downstream evaluation configuration."""

    max_feature_batches: int | None
    linear_probe: LinearProbeConfig
    knn: KNNConfig


class AnalysisConfig(TypedDict):
    """Normalized embedding-analysis configuration."""

    projection_method: ProjectionMethod
    num_query_images: int
    num_neighbors: int


class RootConfig(TypedDict):
    """Normalized root configuration shared by scripts, docs, and tests."""

    seed: int
    dataset: DatasetConfig
    ssl: SSLConfig
    optimizer: OptimizerConfig
    trainer: TrainerConfig
    evaluation: EvaluationConfig
    analysis: AnalysisConfig


def normalize_root_config(raw_config: Mapping[str, Any]) -> RootConfig:
    """Normalize and validate the external YAML config structure.

    Args:
        raw_config: Parsed YAML mapping before default expansion.

    Returns:
        Normalized configuration with concrete defaults for every supported
        section.

    Raises:
        ValueError: If required sections or supported values are invalid.

    Parameters
    ----------
    raw_config : Mapping[str, Any]
        Input value for ``raw_config``.

    Returns
    -------
    result : RootConfig
        Return value produced by the function.
    """
    seed = int(raw_config.get("seed", 7))
    dataset = _normalize_dataset_config(raw_config.get("dataset"))
    ssl = _normalize_ssl_config(raw_config.get("ssl"), dataset)
    optimizer = _normalize_optimizer_config(raw_config.get("optimizer"))
    trainer = _normalize_trainer_config(raw_config.get("trainer"))
    evaluation = _normalize_evaluation_config(raw_config.get("evaluation"))
    analysis = _normalize_analysis_config(raw_config.get("analysis"))
    return {
        "seed": seed,
        "dataset": dataset,
        "ssl": ssl,
        "optimizer": optimizer,
        "trainer": trainer,
        "evaluation": evaluation,
        "analysis": analysis,
    }


def _normalize_dataset_config(raw_dataset_config: Any) -> DatasetConfig:
    """Handle the internal normalize dataset config operation.

    Parameters
    ----------
    raw_dataset_config : Any
        Input value for ``raw_dataset_config``.

    Returns
    -------
    result : DatasetConfig
        Return value produced by the function.
    """
    dataset_config = _as_mapping(raw_dataset_config, section_name="dataset")
    dataset_name = str(dataset_config.get("name", "")).strip().lower()
    if dataset_name not in _SUPPORTED_DATASETS:
        supported = ", ".join(sorted(_SUPPORTED_DATASETS))
        raise ValueError(f"dataset.name must be one of: {supported}")

    dataset_defaults = _SUPPORTED_DATASETS[cast(DatasetName, dataset_name)]
    batch_size = int(dataset_config.get("batch_size", 256))
    eval_batch_size = int(dataset_config.get("eval_batch_size", batch_size))
    return {
        "name": cast(DatasetName, dataset_name),
        "data_dir": str(dataset_config.get("data_dir", "data")),
        "image_size": int(dataset_config.get("image_size", dataset_defaults["image_size"])),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": int(dataset_config.get("num_workers", 4)),
        "download": bool(dataset_config.get("download", True)),
        "use_fake_data": bool(dataset_config.get("use_fake_data", False)),
        "fake_data_size": int(dataset_config.get("fake_data_size", 128)),
        "pretrain_split": str(dataset_config.get("pretrain_split", dataset_defaults["pretrain_split"])),
        "train_split": str(dataset_config.get("train_split", dataset_defaults["train_split"])),
        "test_split": str(dataset_config.get("test_split", dataset_defaults["test_split"])),
    }


def _normalize_ssl_config(raw_ssl_config: Any, dataset_config: DatasetConfig) -> SSLConfig:
    """Handle the internal normalize ssl config operation.

    Parameters
    ----------
    raw_ssl_config : Any
        Input value for ``raw_ssl_config``.
    dataset_config : DatasetConfig
        Input value for ``dataset_config``.

    Returns
    -------
    result : SSLConfig
        Return value produced by the function.
    """
    ssl_config = _as_mapping(raw_ssl_config, section_name="ssl")
    method_name = str(ssl_config.get("method", "")).strip().lower()
    image_size = dataset_config["image_size"]

    if method_name == "mae":
        hidden_dim = int(ssl_config.get("encoder_hidden_dim", 256))
        return {
            "method": "mae",
            "patch_size": int(ssl_config.get("patch_size", 4)),
            "mask_ratio": float(ssl_config.get("mask_ratio", 0.75)),
            "min_scale": float(ssl_config.get("min_scale", 0.2)),
            "encoder_depth": int(ssl_config.get("encoder_depth", 6)),
            "encoder_num_heads": int(ssl_config.get("encoder_num_heads", 8)),
            "encoder_hidden_dim": hidden_dim,
            "encoder_mlp_dim": int(ssl_config.get("encoder_mlp_dim", hidden_dim * 4)),
            "decoder_depth": int(ssl_config.get("decoder_depth", 2)),
            "decoder_num_heads": int(ssl_config.get("decoder_num_heads", 8)),
            "decoder_hidden_dim": int(ssl_config.get("decoder_hidden_dim", 128)),
            "decoder_mlp_dim": int(ssl_config.get("decoder_mlp_dim", 256)),
            "normalize_pixel_targets": bool(ssl_config.get("normalize_pixel_targets", True)),
            "embedding_pool": cast(EmbeddingPool, str(ssl_config.get("embedding_pool", "cls")).lower()),
            "dropout": float(ssl_config.get("dropout", 0.0)),
            "attention_dropout": float(ssl_config.get("attention_dropout", 0.0)),
        }

    if method_name == "dino":
        gaussian_blur = [float(value) for value in ssl_config.get("gaussian_blur", [1.0, 0.1, 0.5])]
        if len(gaussian_blur) != 3:
            raise ValueError("ssl.gaussian_blur must contain three values for DINO tutorials.")
        return {
            "method": "dino",
            "backbone": str(ssl_config.get("backbone", "resnet18")),
            "global_crop_size": int(ssl_config.get("global_crop_size", image_size)),
            "global_crop_scale": [float(value) for value in ssl_config.get("global_crop_scale", [0.5, 1.0])],
            "local_crop_size": int(ssl_config.get("local_crop_size", max(image_size // 2, 32))),
            "local_crop_scale": [float(value) for value in ssl_config.get("local_crop_scale", [0.2, 0.5])],
            "n_local_views": int(ssl_config.get("n_local_views", 4)),
            "projection_hidden_dim": int(ssl_config.get("projection_hidden_dim", 1024)),
            "projection_bottleneck_dim": int(ssl_config.get("projection_bottleneck_dim", 256)),
            "projection_output_dim": int(ssl_config.get("projection_output_dim", 4096)),
            "projection_batch_norm": bool(ssl_config.get("projection_batch_norm", True)),
            "freeze_last_layer_epochs": int(ssl_config.get("freeze_last_layer_epochs", 1)),
            "norm_last_layer": bool(ssl_config.get("norm_last_layer", True)),
            "teacher_momentum_start": float(ssl_config.get("teacher_momentum_start", 0.996)),
            "teacher_momentum_end": float(ssl_config.get("teacher_momentum_end", 1.0)),
            "warmup_teacher_temp": float(ssl_config.get("warmup_teacher_temp", 0.04)),
            "teacher_temp": float(ssl_config.get("teacher_temp", 0.04)),
            "warmup_teacher_temp_epochs": int(ssl_config.get("warmup_teacher_temp_epochs", 10)),
            "student_temp": float(ssl_config.get("student_temp", 0.1)),
            "center_momentum": float(ssl_config.get("center_momentum", 0.9)),
            "color_jitter_strength": float(ssl_config.get("color_jitter_strength", 0.5)),
            "random_gray_scale": float(ssl_config.get("random_gray_scale", 0.2)),
            "gaussian_blur": gaussian_blur,
            "solarization_prob": float(ssl_config.get("solarization_prob", 0.0)),
        }

    raise ValueError("ssl.method must be either 'mae' or 'dino'.")


def _normalize_optimizer_config(raw_optimizer_config: Any) -> OptimizerConfig:
    """Handle the internal normalize optimizer config operation.

    Parameters
    ----------
    raw_optimizer_config : Any
        Input value for ``raw_optimizer_config``.

    Returns
    -------
    result : OptimizerConfig
        Return value produced by the function.
    """
    optimizer_config = _as_mapping(raw_optimizer_config, section_name="optimizer", allow_empty=True)
    learning_rate = float(optimizer_config.get("lr", 1e-3))
    weight_decay = float(optimizer_config.get("weight_decay", 1e-4))
    scheduler_config = _normalize_scheduler_config(optimizer_config.get("scheduler"))
    return {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "scheduler": scheduler_config,
    }


def _normalize_scheduler_config(raw_scheduler_config: Any) -> SchedulerConfig | None:
    """Handle the internal normalize scheduler config operation.

    Parameters
    ----------
    raw_scheduler_config : Any
        Input value for ``raw_scheduler_config``.

    Returns
    -------
    result : SchedulerConfig | None
        Return value produced by the function.
    """
    if raw_scheduler_config is None:
        return None

    scheduler_config = _as_mapping(raw_scheduler_config, section_name="optimizer.scheduler", allow_empty=True)
    if not bool(scheduler_config.get("enabled", False)):
        return None

    scheduler_name = str(scheduler_config.get("name", "cosine_with_warmup"))
    if scheduler_name != "cosine_with_warmup":
        raise ValueError("optimizer.scheduler.name must be 'cosine_with_warmup'.")

    interval = str(scheduler_config.get("interval", "step"))
    if interval != "step":
        raise ValueError("optimizer.scheduler.interval must be 'step'.")

    return {
        "enabled": True,
        "name": "cosine_with_warmup",
        "warmup_epochs": float(scheduler_config.get("warmup_epochs", 0.0)),
        "min_lr": float(scheduler_config.get("min_lr", 0.0)),
        "interval": "step",
    }


def _normalize_trainer_config(raw_trainer_config: Any) -> TrainerConfig:
    """Handle the internal normalize trainer config operation.

    Parameters
    ----------
    raw_trainer_config : Any
        Input value for ``raw_trainer_config``.

    Returns
    -------
    result : TrainerConfig
        Return value produced by the function.
    """
    trainer_config = _as_mapping(raw_trainer_config, section_name="trainer", allow_empty=True)
    progress_bar = str(trainer_config.get("progress_bar", "auto")).strip().lower()
    if progress_bar not in {"auto", "rich", "tqdm"}:
        raise ValueError("trainer.progress_bar must be one of: auto, rich, tqdm")
    return {
        "accelerator": trainer_config.get("accelerator", "auto"),
        "devices": trainer_config.get("devices", "auto"),
        "max_epochs": int(trainer_config.get("max_epochs", 10)),
        "limit_train_batches": trainer_config.get("limit_train_batches", 1.0),
        "log_every_n_steps": int(trainer_config.get("log_every_n_steps", 10)),
        "enable_logging": bool(trainer_config.get("enable_logging", True)),
        "enable_progress_bar": bool(trainer_config.get("enable_progress_bar", True)),
        "progress_bar": cast(ProgressBarStyle, progress_bar),
        "progress_refresh_rate": int(trainer_config.get("progress_refresh_rate", 1)),
        "deterministic": bool(trainer_config.get("deterministic", False)),
        "fast_dev_run": bool(trainer_config.get("fast_dev_run", False)),
    }


def _normalize_evaluation_config(raw_evaluation_config: Any) -> EvaluationConfig:
    """Handle the internal normalize evaluation config operation.

    Parameters
    ----------
    raw_evaluation_config : Any
        Input value for ``raw_evaluation_config``.

    Returns
    -------
    result : EvaluationConfig
        Return value produced by the function.
    """
    evaluation_config = _as_mapping(raw_evaluation_config, section_name="evaluation", allow_empty=True)
    raw_max_batches = evaluation_config.get("max_feature_batches")
    max_feature_batches = None if raw_max_batches is None else int(raw_max_batches)
    linear_probe_config = _as_mapping(evaluation_config.get("linear_probe"), section_name="evaluation.linear_probe", allow_empty=True)
    knn_config = _as_mapping(evaluation_config.get("knn"), section_name="evaluation.knn", allow_empty=True)
    return {
        "max_feature_batches": max_feature_batches,
        "linear_probe": {
            "epochs": int(linear_probe_config.get("epochs", 25)),
            "lr": float(linear_probe_config.get("lr", 1e-2)),
            "weight_decay": float(linear_probe_config.get("weight_decay", 0.0)),
            "batch_size": int(linear_probe_config.get("batch_size", 256)),
        },
        "knn": {
            "k": int(knn_config.get("k", 5)),
            "chunk_size": int(knn_config.get("chunk_size", 512)),
        },
    }


def _normalize_analysis_config(raw_analysis_config: Any) -> AnalysisConfig:
    """Handle the internal normalize analysis config operation.

    Parameters
    ----------
    raw_analysis_config : Any
        Input value for ``raw_analysis_config``.

    Returns
    -------
    result : AnalysisConfig
        Return value produced by the function.
    """
    analysis_config = _as_mapping(raw_analysis_config, section_name="analysis", allow_empty=True)
    projection_method = str(analysis_config.get("projection_method", "pca")).strip().lower()
    if projection_method not in {"pca", "tsne"}:
        raise ValueError("analysis.projection_method must be either 'pca' or 'tsne'.")
    return {
        "projection_method": cast(ProjectionMethod, projection_method),
        "num_query_images": int(analysis_config.get("num_query_images", 8)),
        "num_neighbors": int(analysis_config.get("num_neighbors", 5)),
    }


def _as_mapping(value: Any, *, section_name: str, allow_empty: bool = False) -> Mapping[str, Any]:
    """Handle the internal as mapping operation.

    Parameters
    ----------
    value : Any
        Input value for ``value``.
    section_name : str
        Input value for ``section_name``.
    allow_empty : bool
        Input value for ``allow_empty``.

    Returns
    -------
    result : Mapping[str, Any]
        Return value produced by the function.
    """
    if value is None and allow_empty:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{section_name}' must be a mapping.")
    return value
