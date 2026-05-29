"""CLI for training tutorial SSL models from YAML configs."""

from __future__ import annotations

import argparse
import sys
import torch

torch.set_float32_matmul_precision('medium')

from textwrap import dedent
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from fomocid.config_types import RootConfig, TrainerConfig
from fomocid.data import create_datamodule
from fomocid.ssl import build_ssl_module, get_ssl_method_name
from fomocid.utils import create_run_dir, load_config, save_json, save_yaml, seed_everything


CONFIG_GUIDE = dedent(
    """
    Config guide:

    - seed:
      Global random seed for PyTorch, NumPy, and data-loader workers.

    - dataset:
      name: symbolic benchmark name. Phase 1 supports `cifar10`, `mnist`, and `stl10`.
      data_dir: where datasets are stored or downloaded.
      image_size: input resolution used for both pretraining and evaluation transforms.
      batch_size / eval_batch_size: batch sizes for SSL training and downstream analysis.
      num_workers: number of data-loader workers.
      download: whether benchmark data should be downloaded automatically.
      use_fake_data / fake_data_size: smoke-test switches for CI and quick debugging.
      pretrain_split / train_split / test_split: split policy for SSL pretraining and evaluation.

    - ssl:
      method: `mae` or `dino`.
      MAE-specific keys:
        patch_size, mask_ratio, min_scale, encoder_depth, encoder_num_heads,
        encoder_hidden_dim, encoder_mlp_dim, decoder_depth, decoder_num_heads,
        decoder_hidden_dim, decoder_mlp_dim, normalize_pixel_targets,
        embedding_pool (`cls` or `mean`) for downstream feature extraction,
        dropout, attention_dropout.
      DINO-specific keys:
        backbone, global_crop_size, global_crop_scale, local_crop_size,
        local_crop_scale, n_local_views, projection_hidden_dim,
        projection_bottleneck_dim, projection_output_dim, projection_batch_norm,
        freeze_last_layer_epochs, norm_last_layer, teacher_momentum_start,
        teacher_momentum_end, warmup_teacher_temp, teacher_temp,
        warmup_teacher_temp_epochs, student_temp, center_momentum,
        color_jitter_strength, random_gray_scale, gaussian_blur,
        solarization_prob.

    - optimizer:
      lr: peak learning rate used by AdamW after warmup completes.
      weight_decay: AdamW weight decay.
      scheduler:
        enabled: toggles scheduler support while preserving backward compatibility.
        name: phase 1 supports `cosine_with_warmup`.
        warmup_epochs: linearly ramps to `optimizer.lr`; internally converted to steps.
        min_lr: floor used by the cosine decay phase after warmup.
        interval: fixed to `step` in phase 1.

    - trainer:
      accelerator / devices: hardware selection passed through to Lightning.
      max_epochs: number of epochs.
      limit_train_batches: useful for smoke runs and debugging.
      log_every_n_steps: logging cadence.
      enable_logging: when false, disables Lightning experiment loggers.
      enable_progress_bar: when false, hides the Lightning terminal progress bar.
      progress_bar: `auto`, `rich`, or `tqdm`; `auto` uses `tqdm` in notebooks
        and `rich` in terminal sessions.
      progress_refresh_rate: refresh interval for the tqdm progress bar.
      deterministic: reproducibility toggle.
      fast_dev_run: Lightning one-step sanity mode.

    - evaluation:
      max_feature_batches: optional cap for feature extraction during evaluation.
      linear_probe: epochs, lr, weight_decay, batch_size for the frozen probe.
      knn: k and chunk_size for cosine-similarity retrieval evaluation.

    - analysis:
      projection_method: `pca` or `tsne` for the embedding figure.
      num_query_images: number of query examples rendered in the retrieval grid.
      num_neighbors: number of nearest neighbors shown per query.

    Training outputs:
    - resolved_config.yaml: exact config used for the run
    - logs/: Lightning CSV logs, only when trainer.enable_logging is true
    - checkpoints/: epoch checkpoints plus last.ckpt
    - train_summary.json: machine-readable paths to the main artifacts
    """
).strip()

CLI_EPILOG = dedent(
    """
    Examples:
      python tutorials/train_ssl.py --config configs/cifar10_mae.yaml --output-dir outputs
      python tutorials/train_ssl.py --config configs/stl10_dino.yaml --output-dir outputs

    Tip:
      Run with --explain-config to print a compact reference for all supported config sections.
    """
).strip()


def _build_trainer(config: RootConfig, run_dir: Path) -> Trainer:
    """Build the Lightning trainer used by tutorial pretraining runs.

    Args:
        config: Normalized root config whose ``trainer`` section controls
            runtime behavior such as epochs, devices, logging, and progress
            bars.
        run_dir: Run directory where checkpoints and logs should be written.

    Returns:
        Configured Lightning trainer.

    Parameters
    ----------
    config : RootConfig
        Input value for ``config``.
    run_dir : Path
        Input value for ``run_dir``.

    Returns
    -------
    result : Trainer
        Return value produced by the function.
    """
    trainer_config = config["trainer"]
    checkpoint_dir = run_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="epoch{epoch:02d}",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
        )
    ]
    enable_logging = bool(trainer_config.get("enable_logging", True))
    enable_progress_bar = bool(trainer_config.get("enable_progress_bar", True))
    if enable_progress_bar:
        callbacks.append(_build_progress_bar_callback(trainer_config))
    logger = CSVLogger(save_dir=str(run_dir), name="logs") if enable_logging else False
    return Trainer(
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=trainer_config.get("devices", "auto"),
        max_epochs=int(trainer_config.get("max_epochs", 10)),
        limit_train_batches=trainer_config.get("limit_train_batches", 1.0),
        log_every_n_steps=int(trainer_config.get("log_every_n_steps", 10)),
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        callbacks=callbacks,
        logger=logger,
        deterministic=bool(trainer_config.get("deterministic", False)),
        fast_dev_run=bool(trainer_config.get("fast_dev_run", False)),
    )


def _build_progress_bar_callback(trainer_config: TrainerConfig) -> Any:
    """Select a progress bar that matches the current execution environment.

    Args:
        trainer_config: Normalized ``trainer`` section containing the
            ``progress_bar`` style and ``progress_refresh_rate``.

    Returns:
        Progress-bar callback instance for Lightning.

    Raises:
        ValueError: If the configured progress bar style is unsupported.

    Parameters
    ----------
    trainer_config : TrainerConfig
        Input value for ``trainer_config``.

    Returns
    -------
    result : Any
        Return value produced by the function.
    """
    progress_bar_style = str(trainer_config.get("progress_bar", "auto")).strip().lower()
    if progress_bar_style == "auto":
        progress_bar_style = "tqdm" if _is_notebook_session() else "rich"

    if progress_bar_style == "rich":
        return RichProgressBar(leave=True)
    if progress_bar_style == "tqdm":
        return TQDMProgressBar(
            refresh_rate=int(trainer_config.get("progress_refresh_rate", 1)),
            leave=True,
        )
    raise ValueError(
        f"Unsupported trainer.progress_bar value: {progress_bar_style}. Supported values: auto, rich, tqdm"
    )


def _is_notebook_session() -> bool:
    """Return whether training is running inside a notebook kernel.

    Parameters
    ----------
    None
        This function takes no explicit input parameters.

    Returns
    -------
    result : bool
        Return value produced by the function.
    """
    return "ipykernel" in sys.modules


def run_training(config_path: str | Path, output_dir: str | Path) -> Path:
    """Train one tutorial SSL run and persist the main artifacts.

    Args:
        config_path: Path to a YAML config describing the ``dataset``, ``ssl``,
            ``optimizer``, and ``trainer`` sections.
        output_dir: Root directory where the timestamped run folder should be
            created.

    Returns:
        Path to the created run directory containing checkpoints, logs, and the
        resolved config snapshot.

    Parameters
    ----------
    config_path : str | Path
        Input value for ``config_path``.
    output_dir : str | Path
        Input value for ``output_dir``.

    Returns
    -------
    result : Path
        Return value produced by the function.
    """
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 7)))

    dataset_name = config["dataset"]["name"]
    method_name = get_ssl_method_name(config)
    run_dir = create_run_dir(output_dir, stem=f"{dataset_name}_{method_name}")
    save_yaml(run_dir / "resolved_config.yaml", config)

    datamodule = create_datamodule(config)
    model = build_ssl_module(config)
    trainer = _build_trainer(config, run_dir)
    trainer.fit(model, datamodule=datamodule)

    artifacts = {
        "run_dir": str(run_dir),
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "last_checkpoint": str(run_dir / "checkpoints" / "last.ckpt"),
    }
    save_json(run_dir / "train_summary.json", artifacts)
    return run_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training CLI.

    Returns:
        Parsed namespace containing config path, output directory, and the
        optional config-explainer flag.

    Parameters
    ----------
    None
        This function takes no explicit input parameters.

    Returns
    -------
    result : argparse.Namespace
        Return value produced by the function.
    """
    parser = argparse.ArgumentParser(
        description="Train a tutorial SSL model from a YAML configuration.",
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config that selects the benchmark, SSL method, optimization settings, and trainer limits.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory where a timestamped run folder will be created.",
    )
    parser.add_argument(
        "--explain-config",
        action="store_true",
        help="Print a compact explanation of the expected config sections and exit.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the training CLI entrypoint.

    Parameters
    ----------
    None
        This function takes no explicit input parameters.

    Returns
    -------
    None
        The function completes in place.
    """
    args = parse_args()
    if args.explain_config:
        print(CONFIG_GUIDE)
        return
    run_dir = run_training(args.config, args.output_dir)
    print(run_dir)


if __name__ == "__main__":
    main()
