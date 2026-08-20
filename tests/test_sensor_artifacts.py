import numpy as np

from scattering_calculator.experimental_conditions.detector import detector_hologram


def _artifact_model(**artifacts):
    model = object.__new__(detector_hologram)
    model.detector_threshold = 1000.0
    model.number_frames = 1
    model.exposure_time = 1.0
    model.hologram = np.zeros((32, 32))
    config = {**detector_hologram.DEFAULT_ARTIFACTS_CONFIG, **artifacts}
    model._apply_artifacts_config(config)
    return model


def test_camera_seed_reuses_the_same_defect_positions():
    first = _artifact_model(
        camera_seed=42, average_hot_pixels=20, average_cold_pixels=10
    )
    second = _artifact_model(
        camera_seed=42, average_hot_pixels=20, average_cold_pixels=10
    )

    map_a = first._camera_defect_map((32, 32))
    map_b = second._camera_defect_map((32, 32))

    assert all(np.array_equal(map_a[key], map_b[key]) for key in ("hot", "cold"))


def test_cosmic_ray_count_uses_one_second_when_exposure_is_missing():
    model = _artifact_model(camera_seed=3, cosmic_rays_per_second=5.0)
    model.exposure_time = None
    rng_a = np.random.default_rng(9)
    rng_b = np.random.default_rng(9)

    model._add_sensor_artifacts(np.zeros((32, 32)), rng_a)
    expected = rng_b.poisson(5.0)

    assert model.cosmic_ray_count == expected
    assert model.cosmic_ray_tracks.shape == (expected, 6)
    if expected:
        assert np.all((model.cosmic_ray_tracks[:, 3] >= 2.0) & (model.cosmic_ray_tracks[:, 3] <= 3.0))


def test_zero_rates_leave_image_unchanged():
    model = _artifact_model(camera_seed=1)
    image = np.arange(64, dtype=float).reshape(8, 8)

    result = model._add_sensor_artifacts(image, np.random.default_rng(2))

    np.testing.assert_array_equal(result, image)


def test_seeded_pixel_baselines_persist_but_exposure_values_oscillate():
    model = _artifact_model(
        camera_seed=77,
        average_hot_pixels=50,
        hot_pixel_value=800.0,
        hot_pixel_value_spread=0.1,
        hot_pixel_temporal_sigma=0.05,
    )
    map_a = model._camera_defect_map((32, 32))
    map_b = model._camera_defect_map((32, 32))
    np.testing.assert_array_equal(
        map_a["hot_baseline_values"], map_b["hot_baseline_values"]
    )

    first = model._add_sensor_artifacts(
        np.zeros((32, 32)), np.random.default_rng(10)
    )
    second = model._add_sensor_artifacts(
        np.zeros((32, 32)), np.random.default_rng(11)
    )
    assert np.any(first[map_a["hot"]] != second[map_a["hot"]])


def test_cosmic_rays_vary_aspect_ratio_and_peak_within_one_image():
    model = _artifact_model(
        cosmic_rays_per_second=100.0,
        cosmic_ray_value=700.0,
        cosmic_ray_value_spread=0.2,
        cosmic_ray_length_range=(2.0, 5.0),
        cosmic_ray_aspect_ratio_range=(2.0, 3.0),
    )
    model._add_sensor_artifacts(np.zeros((64, 64)), np.random.default_rng(12))

    assert np.ptp(model.cosmic_ray_tracks[:, 3]) > 0
    assert np.ptp(model.cosmic_ray_tracks[:, 5]) > 0
