"""CLI for probing and visualizing trained tutorial SSL checkpoints."""

from __future__ import annotations

import argparse
import sys
from textwrap import dedent
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from fomocid.config_types import AnalysisConfig, EvaluationConfig, RootConfig
from fomocid.analysis import build_nearest_neighbor_figure, project_embeddings, save_projection_figure
from fomocid.analysis.projections import retrieve_nearest_neighbors
from fomocid.data import create_datamodule
from fomocid.eval import evaluate_embeddings, extract_features, save_embedding_bundle
from fomocid.eval.features import save_metrics
from fomocid.ssl import load_ssl_module
from fomocid.utils import create_run_dir, load_config, save_json, save_yaml, seed_everything


CLI_EPILOG = dedent(
    """
    Evaluation outputs:
      - embeddings.pt: extracted train/test embeddings and labels
      - metrics.json: linear-probe and kNN results
      - projection.png: 2D projection of the test embeddings
      - nearest_neighbors.png: query-to-neighbor retrieval grid

    Example:
      python tutorials/evaluate_representations.py \
        --config configs/cifar10_mae.yaml \
        --checkpoint-path outputs/<run>/checkpoints/last.ckpt \
        --output-dir outputs
    """
).strip()


def _latest_checkpoint(output_dir: Path) -> Path:
    """Return the most recently modified ``last.ckpt`` under an output tree.

    Args:
        output_dir: Root directory containing one or more prior training runs.

    Returns:
        Path to the newest checkpoint file.

    Raises:
        FileNotFoundError: If no ``last.ckpt`` file can be found.
    """
    candidates = sorted(output_dir.glob("**/checkpoints/last.ckpt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint-path or point --output-dir to a directory containing a prior run."
        )
    return candidates[-1]


def _resolve_eval_dir(output_dir: Path, checkpoint_path: Path) -> Path:
    """Choose or create the directory where evaluation artifacts are written.

    Args:
        output_dir: Fallback output root when the checkpoint does not live
            inside a recognizable run directory.
        checkpoint_path: Checkpoint being evaluated.

    Returns:
        Evaluation directory path.
    """
    run_root = checkpoint_path.parent.parent
    if run_root.exists():
        eval_dir = run_root / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir
    return create_run_dir(output_dir, stem="evaluation")


def run_evaluation(
    config_path: str | Path,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
) -> Path:
    """Evaluate a trained SSL checkpoint with probes and visualizations.

    Args:
        config_path: Path to a YAML config describing the dataset,
            evaluation, and analysis defaults.
        output_dir: Root directory used to discover prior runs and to create a
            new evaluation directory when needed.
        checkpoint_path: Optional checkpoint override. When omitted, the newest
            ``last.ckpt`` below ``output_dir`` is used.

    Returns:
        Path to the evaluation directory containing embeddings, metrics, and
        figures.
    """
    config = load_config(config_path)
    seed_everything(int(config.get("seed", 7)))

    output_dir = Path(output_dir)
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else _latest_checkpoint(output_dir)
    eval_dir = _resolve_eval_dir(output_dir, resolved_checkpoint)
    save_yaml(eval_dir / "resolved_config.yaml", config)

    datamodule = create_datamodule(config)
    datamodule.setup()
    model = load_ssl_module(str(resolved_checkpoint), config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analysis_config: AnalysisConfig = config.get("analysis", {})
    eval_config: EvaluationConfig = config.get("evaluation", {})

    train_features = extract_features(
        model,
        datamodule.labeled_train_dataloader(),
        device=device,
        max_batches=eval_config.get("max_feature_batches"),
    )
    test_features = extract_features(
        model,
        datamodule.test_dataloader(),
        device=device,
        max_batches=eval_config.get("max_feature_batches"),
    )

    embeddings_path = save_embedding_bundle(eval_dir / "embeddings.pt", train_features, test_features)
    metrics = evaluate_embeddings(
        train_features,
        test_features,
        num_classes=datamodule.num_classes,
        linear_probe_config=eval_config.get("linear_probe", {}),
        knn_config=eval_config.get("knn", {}),
        device=device,
    )

    projected, used_method = project_embeddings(
        test_features["embeddings"],
        method=analysis_config.get("projection_method", "pca"),
    )
    projection_path = save_projection_figure(
        projected,
        test_features["labels"],
        eval_dir / "projection.png",
        title=f"{config['dataset']['name']} {used_method.upper()} projection",
    )

    num_queries = int(analysis_config.get("num_query_images", 8))
    num_neighbors = int(analysis_config.get("num_neighbors", 5))
    query_images = test_features["images"][:num_queries]
    query_embeddings = test_features["embeddings"][:num_queries]
    query_labels = test_features["labels"][:num_queries]
    neighbor_indices, _scores = retrieve_nearest_neighbors(
        train_features["embeddings"],
        query_embeddings,
        k=num_neighbors,
    )
    nearest_neighbor_path = build_nearest_neighbor_figure(
        train_features["images"],
        query_images,
        neighbor_indices,
        eval_dir / "nearest_neighbors.png",
        mean=datamodule.mean,
        std=datamodule.std,
        query_labels=query_labels,
        reference_labels=train_features["labels"],
        label_names=datamodule.class_names,
    )

    metrics["checkpoint_path"] = str(resolved_checkpoint)
    metrics["projection_method"] = used_method
    metrics["embeddings_path"] = str(embeddings_path)
    metrics["projection_figure"] = str(projection_path)
    metrics["nearest_neighbor_figure"] = str(nearest_neighbor_path)
    metrics_path = save_metrics(eval_dir / "metrics.json", metrics)
    save_json(
        eval_dir / "evaluation_summary.json",
        {
            "evaluation_dir": str(eval_dir),
            "checkpoint_path": str(resolved_checkpoint),
            "metrics_path": str(metrics_path),
        },
    )
    return eval_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation CLI.

    Returns:
        Parsed namespace containing config path, output root, and optional
        checkpoint override.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SSL checkpoint with probing, retrieval, and embedding analysis outputs.",
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="YAML config describing the dataset resolution and evaluation defaults.")
    parser.add_argument("--output-dir", required=True, help="Root output directory used to locate previous runs and store evaluation artifacts.")
    parser.add_argument(
        "--checkpoint-path",
        help="Checkpoint to evaluate. If omitted, the script picks the newest last.ckpt found under output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the evaluation CLI entrypoint."""
    args = parse_args()
    eval_dir = run_evaluation(args.config, args.output_dir, checkpoint_path=args.checkpoint_path)
    print(eval_dir)


if __name__ == "__main__":
    main()
