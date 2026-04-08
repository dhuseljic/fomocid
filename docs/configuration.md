# Configuration Reference

All tutorial YAML files normalize to the same top-level structure:

- `seed`
- `dataset`
- `ssl`
- `optimizer`
- `trainer`
- `evaluation`
- `analysis`

The loader in `fomocid.utils.load_config()` validates this shape and fills in defaults so scripts, tests, notebooks, and the API all see the same normalized config.

## `seed`

- `seed`: integer random seed used for PyTorch, NumPy, and worker seeding. Default: `7`.

## `dataset`

- `name`: benchmark name. Supported values: `cifar10`, `mnist`, `stl10`.
- `data_dir`: dataset root or download location. Default: `data`.
- `image_size`: input resolution used for both training and evaluation transforms. Defaults: `32` for `cifar10`, `28` for `mnist`, `96` for `stl10`.
- `batch_size`: SSL training batch size. Default: `256`.
- `eval_batch_size`: evaluation batch size. Default: `dataset.batch_size`.
- `num_workers`: dataloader worker count. Default: `4`.
- `download`: whether missing benchmark data should be downloaded. Default: `true`.
- `use_fake_data`: switch to `torchvision.datasets.FakeData` for smoke tests and quick debugging. Default: `false`.
- `fake_data_size`: dataset size when `use_fake_data=true`. Default: `128`.
- `pretrain_split`: split used for SSL pretraining. Defaults: `train` for `cifar10` and `mnist`, `train+unlabeled` for `stl10`.
- `train_split`: labeled split used for linear probing and retrieval bank construction. Default: `train`.
- `test_split`: held-out split used for evaluation. Default: `test`.

## `ssl`

`ssl.method` chooses the active tutorial family.

### Common selector

- `method`: `mae` or `dino`.

### MAE keys

- `patch_size`: square patch size in pixels. Default: `4`.
- `mask_ratio`: fraction of patches hidden from the encoder. Default: `0.75`.
- `min_scale`: minimum crop scale for `MAETransform`. Default: `0.2`.
- `encoder_depth`: transformer encoder depth. Default: `6`.
- `encoder_num_heads`: encoder attention heads. Default: `8`.
- `encoder_hidden_dim`: encoder hidden width. Default: `256`.
- `encoder_mlp_dim`: encoder MLP width. Default: `encoder_hidden_dim * 4`.
- `decoder_depth`: decoder transformer depth. Default: `2`.
- `decoder_num_heads`: decoder attention heads. Default: `8`.
- `decoder_hidden_dim`: decoder hidden width. Default: `128`.
- `decoder_mlp_dim`: decoder MLP width. Default: `256`.
- `normalize_pixel_targets`: whether masked reconstruction targets are standardized patch-wise before loss computation. Default: `true`.
- `embedding_pool`: image-level feature pooling for downstream evaluation. Supported values: `cls`, `mean`. Default: `cls`.
- `dropout`: transformer dropout applied to encoder and decoder blocks. Default: `0.0`.
- `attention_dropout`: attention dropout for the encoder ViT. Default: `0.0`.

### DINO keys

- `backbone`: student/teacher encoder backbone. Current supported value: `resnet18`.
- `global_crop_size`: size of the two global crops. Default: `dataset.image_size`.
- `global_crop_scale`: scale range for global crops. Default: `[0.5, 1.0]`.
- `local_crop_size`: size of the local crops. Default: `max(dataset.image_size // 2, 32)`.
- `local_crop_scale`: scale range for local crops. Default: `[0.2, 0.5]`.
- `n_local_views`: number of local crops added to the two global crops. Default: `4`.
- `projection_hidden_dim`: DINO head hidden width. Default: `1024`.
- `projection_bottleneck_dim`: DINO head bottleneck width. Default: `256`.
- `projection_output_dim`: DINO head output width. Default: `4096`.
- `projection_batch_norm`: whether to use batch norm inside the DINO head. Default: `true`.
- `freeze_last_layer_epochs`: number of epochs to cancel gradients for the last DINO layer. Default: `1`.
- `norm_last_layer`: whether to normalize the last DINO layer. Default: `true`.
- `teacher_momentum_start`: initial EMA momentum for teacher updates. Default: `0.996`.
- `teacher_momentum_end`: final EMA momentum reached by the end of training. Default: `1.0`.
- `warmup_teacher_temp`: initial teacher temperature during warmup. Default: `0.04`.
- `teacher_temp`: steady-state teacher temperature. Default: `0.04`.
- `warmup_teacher_temp_epochs`: number of warmup epochs for teacher temperature. Default: `10`.
- `student_temp`: student temperature. Default: `0.1`.
- `center_momentum`: center EMA momentum in the DINO loss. Default: `0.9`.
- `color_jitter_strength`: color jitter strength for DINO augmentations. Default: `0.5`.
- `random_gray_scale`: grayscale augmentation probability. Default: `0.2`.
- `gaussian_blur`: three blur strengths for global-0, global-1, and local views. Must contain exactly three values. Default: `[1.0, 0.1, 0.5]`.
- `solarization_prob`: solarization probability applied to the second global crop path. Default: `0.0`.

## `optimizer`

- `lr`: peak AdamW learning rate. Default: `1e-3`.
- `weight_decay`: AdamW weight decay. Default: `1e-4`.
- `scheduler`: optional subsection. Omit it, or set `enabled: false`, to disable scheduling.

### `optimizer.scheduler`

- `enabled`: toggles scheduler support.
- `name`: current supported value: `cosine_with_warmup`.
- `warmup_epochs`: linear warmup duration expressed in epochs. Default: `0.0`.
- `min_lr`: cosine decay floor. Default: `0.0`.
- `interval`: fixed to `step`.

## `trainer`

- `accelerator`: Lightning accelerator value. Default: `auto`.
- `devices`: Lightning device selection. Default: `auto`.
- `max_epochs`: number of training epochs. Default: `10`.
- `limit_train_batches`: useful for smoke runs and debugging. Default: `1.0`.
- `log_every_n_steps`: training log cadence. Default: `10`.
- `enable_logging`: toggles CSV logging. Default: `true`.
- `enable_progress_bar`: toggles progress-bar callbacks. Default: `true`.
- `progress_bar`: `auto`, `rich`, or `tqdm`. Default: `auto`.
- `progress_refresh_rate`: refresh interval for the tqdm progress bar. Default: `1`.
- `deterministic`: Lightning deterministic toggle. Default: `false`.
- `fast_dev_run`: Lightning one-step sanity mode. Default: `false`.

## `evaluation`

- `max_feature_batches`: optional cap for feature extraction batches. Default: unset.
- `linear_probe`: subsection for frozen-feature classifier training.
- `knn`: subsection for cosine-similarity retrieval evaluation.

### `evaluation.linear_probe`

- `epochs`: number of linear-probe epochs. Default: `25`.
- `lr`: linear-probe learning rate. Default: `1e-2`.
- `weight_decay`: linear-probe weight decay. Default: `0.0`.
- `batch_size`: linear-probe batch size. Default: `256`.

### `evaluation.knn`

- `k`: number of nearest neighbors. Default: `5`.
- `chunk_size`: query batch size for similarity computation. Default: `512`.

## `analysis`

- `projection_method`: embedding projection backend used for `projection.png`. Supported values: `pca`, `tsne`. Default: `pca`.
- `num_query_images`: number of test images shown in the nearest-neighbor figure. Default: `8`.
- `num_neighbors`: number of retrieved neighbors per query. Default: `5`.

## Shipped configs

- `configs/cifar10_mae.yaml`: full CIFAR-10 MAE tutorial run.
- `configs/mnist_mae.yaml`: fast real-dataset MAE run on MNIST.
- `configs/stl10_dino.yaml`: full STL-10 DINO tutorial run.
- `configs/smoke_cifar10_mae.yaml`: fake-data MAE smoke config for tests and CI.
- `configs/smoke_mnist_mae.yaml`: fake-data MNIST-shaped MAE smoke config for tests and CI.
- `configs/smoke_stl10_dino.yaml`: fake-data DINO smoke config for tests and CI.
