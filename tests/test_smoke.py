from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
import copy
from PIL import Image

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fomocid.data import create_datamodule
from fomocid.ssl.common import build_optimizer, build_scheduler, resolve_optimizer_config, warmup_cosine_factor
from fomocid.utils import (
    build_dataset_preview,
    build_mae_mask_preview,
    build_pretrain_view_preview,
    load_config,
    save_yaml,
    load_config_text,
    summarize_ssl_batch_shapes,
)
from tutorials.evaluate_representations import run_evaluation
from tutorials.train_ssl import run_training


class SmokeTests(unittest.TestCase):
    def test_cifar10_mae_config_builds_datamodule(self) -> None:
        config = load_config(Path("configs/smoke_cifar10_mae.yaml"))
        datamodule = create_datamodule(config)
        datamodule.setup()

        ssl_batch = next(iter(datamodule.ssl_train_dataloader()))
        eval_batch = next(iter(datamodule.test_dataloader()))

        self.assertEqual(config["dataset"]["name"], "cifar10")
        self.assertEqual(len(ssl_batch[0]), 1)
        self.assertEqual(ssl_batch[0][0].shape[0], config["dataset"]["batch_size"])
        self.assertEqual(eval_batch[0].shape[0], config["dataset"]["eval_batch_size"])

    def test_stl10_dino_config_builds_datamodule(self) -> None:
        config = load_config(Path("configs/smoke_stl10_dino.yaml"))
        datamodule = create_datamodule(config)
        datamodule.setup()

        ssl_batch = next(iter(datamodule.ssl_train_dataloader()))

        self.assertEqual(config["dataset"]["name"], "stl10")
        self.assertEqual(len(ssl_batch[0]), 4)
        self.assertEqual(ssl_batch[0][0].shape[0], config["dataset"]["batch_size"])

    def test_mnist_mae_config_builds_datamodule(self) -> None:
        config = load_config(Path("configs/smoke_mnist_mae.yaml"))
        datamodule = create_datamodule(config)
        datamodule.setup()

        ssl_batch = next(iter(datamodule.ssl_train_dataloader()))
        eval_batch = next(iter(datamodule.test_dataloader()))

        self.assertEqual(config["dataset"]["name"], "mnist")
        self.assertEqual(len(ssl_batch[0]), 1)
        self.assertEqual(ssl_batch[0][0].shape[1], 3)
        self.assertEqual(eval_batch[0].shape[1], 3)

    def test_cifar10_mae_training_and_evaluation_smoke(self) -> None:
        config_path = Path("configs/smoke_cifar10_mae.yaml")
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_training(config_path, tmp_dir)
            checkpoint_path = run_dir / "checkpoints" / "last.ckpt"

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue((run_dir / "resolved_config.yaml").exists())
            self.assertTrue((run_dir / "train_summary.json").exists())

            eval_dir = run_evaluation(config_path, tmp_dir, checkpoint_path=checkpoint_path)

            self.assertTrue((eval_dir / "embeddings.pt").exists())
            self.assertTrue((eval_dir / "metrics.json").exists())
            self.assertTrue((eval_dir / "projection.png").exists())
            self.assertTrue((eval_dir / "nearest_neighbors.png").exists())
            self.assertTrue((eval_dir / "evaluation_summary.json").exists())

    def test_optimizer_config_without_scheduler_uses_plain_adamw(self) -> None:
        base_config = load_config(Path("configs/smoke_cifar10_mae.yaml"))
        config = copy.deepcopy(base_config)
        config["optimizer"].pop("scheduler")

        optimizer_config = resolve_optimizer_config(config)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = build_optimizer([parameter], optimizer_config)
        scheduler = build_scheduler(optimizer, optimizer_config, total_steps=4, max_epochs=1)

        self.assertIsNone(optimizer_config["scheduler"])
        self.assertIsNone(scheduler)

    def test_cosine_scheduler_reaches_peak_and_min_lr(self) -> None:
        config = load_config(Path("configs/smoke_cifar10_mae.yaml"))
        config = copy.deepcopy(config)
        config["optimizer"]["scheduler"]["warmup_epochs"] = 1.0
        optimizer_config = resolve_optimizer_config(config)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = build_optimizer([parameter], optimizer_config)
        scheduler = build_scheduler(optimizer, optimizer_config, total_steps=8, max_epochs=4)

        self.assertIsNotNone(scheduler)
        learning_rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(7):
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        self.assertLess(learning_rates[0], optimizer_config["lr"])
        self.assertAlmostEqual(max(learning_rates), optimizer_config["lr"], places=7)
        self.assertAlmostEqual(learning_rates[-1], optimizer_config["scheduler"]["min_lr"], places=7)

    def test_warmup_cosine_factor_edge_cases(self) -> None:
        self.assertAlmostEqual(warmup_cosine_factor(0, total_steps=4, warmup_steps=0, min_lr_scale=1.0), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(1, total_steps=2, warmup_steps=4, min_lr_scale=0.2), 1.0)
        self.assertAlmostEqual(warmup_cosine_factor(3, total_steps=4, warmup_steps=1, min_lr_scale=0.1), 0.1)

    def test_cifar10_mae_training_without_logging(self) -> None:
        base_config = load_config(Path("configs/smoke_cifar10_mae.yaml"))
        config = copy.deepcopy(base_config)
        config["trainer"]["enable_logging"] = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "smoke_cifar10_mae_no_logging.yaml"
            save_yaml(config_path, config)
            run_dir = run_training(config_path, tmp_dir)

            self.assertTrue((run_dir / "checkpoints" / "last.ckpt").exists())
            self.assertTrue((run_dir / "train_summary.json").exists())
            self.assertFalse((run_dir / "logs").exists())

    def test_cifar10_mae_training_without_progress_bar(self) -> None:
        base_config = load_config(Path("configs/smoke_cifar10_mae.yaml"))
        config = copy.deepcopy(base_config)
        config["trainer"]["enable_progress_bar"] = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "smoke_cifar10_mae_no_progress.yaml"
            save_yaml(config_path, config)
            run_dir = run_training(config_path, tmp_dir)

            self.assertTrue((run_dir / "checkpoints" / "last.ckpt").exists())
            self.assertTrue((run_dir / "train_summary.json").exists())

    def test_stl10_dino_training_and_evaluation_smoke(self) -> None:
        config_path = Path("configs/smoke_stl10_dino.yaml")
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_training(config_path, tmp_dir)
            checkpoint_path = run_dir / "checkpoints" / "last.ckpt"

            self.assertTrue(checkpoint_path.exists())
            eval_dir = run_evaluation(config_path, tmp_dir, checkpoint_path=checkpoint_path)

            self.assertTrue((eval_dir / "embeddings.pt").exists())
            self.assertTrue((eval_dir / "metrics.json").exists())
            self.assertTrue((eval_dir / "projection.png").exists())
            self.assertTrue((eval_dir / "nearest_neighbors.png").exists())

    def test_notebook_helpers_for_cifar10_mae(self) -> None:
        config_path = Path("configs/smoke_cifar10_mae.yaml")
        dataset_preview = build_dataset_preview(config_path, split="train", limit=8, samples_per_class=1)
        mask_preview = build_mae_mask_preview(config_path, split="train", index=0)
        view_preview = build_pretrain_view_preview(config_path, split="train", index=0, repeats=3)
        batch_shapes = summarize_ssl_batch_shapes(config_path)
        config_text = load_config_text(config_path)

        self.assertGreater(dataset_preview.size[0], 0)
        self.assertGreater(mask_preview.size[0], 0)
        self.assertGreater(view_preview.size[0], 0)
        self.assertEqual(batch_shapes["num_views"], 1)
        self.assertIn("ssl:", config_text)
        self.assertGreaterEqual(dataset_preview.size[0], 4 * 192)

    def test_notebook_helpers_for_stl10_dino(self) -> None:
        config_path = Path("configs/smoke_stl10_dino.yaml")
        dataset_preview = build_dataset_preview(config_path, split="train", limit=8, samples_per_class=1)
        view_preview = build_pretrain_view_preview(config_path, split="train", index=0)
        batch_shapes = summarize_ssl_batch_shapes(config_path)

        self.assertGreater(dataset_preview.size[0], 0)
        self.assertGreater(view_preview.size[0], 0)
        self.assertEqual(batch_shapes["num_views"], 4)

    def test_build_dataset_preview_respects_display_size(self) -> None:
        config_path = Path("configs/smoke_mnist_mae.yaml")
        small_preview = build_dataset_preview(config_path, split="train", limit=4, samples_per_class=1, display_size=64)
        large_preview = build_dataset_preview(config_path, split="train", limit=4, samples_per_class=1, display_size=160)

        self.assertLess(small_preview.size[0], large_preview.size[0])

    def test_mnist_base_transform_converts_grayscale_to_rgb(self) -> None:
        config = load_config(Path("configs/smoke_mnist_mae.yaml"))
        datamodule = create_datamodule(config)

        grayscale = Image.new("L", (28, 28), color=128)
        converted = datamodule._compose_dataset_transform(None)(grayscale)

        self.assertEqual(converted.mode, "RGB")
