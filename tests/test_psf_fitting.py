"""Prepared PSF regression cases; no real acquisition or training is required."""
import numpy as np
import pytest

from ionrecog.evaluation import match_points
from ionrecog.psf_fitting import PSFConfig, fit_patch, locate_ions, render_expected


def gaussian_scene(shape, positions, sigma=1.3, background=40.0):
    # Independent analytical fixture, with known continuous center coordinates.
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    image = np.full(shape, background, dtype=float)
    for x, y, amplitude in positions:
        image += amplitude * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*sigma*sigma))
    return image


def test_joint_fit_recovers_two_subpixel_overlapping_emitters():
    truth = [[13.3, 14.6], [16.0, 15.1]]
    raw = gaussian_scene((32, 32), [(*truth[0], 650), (*truth[1], 410)])
    result = fit_patch(raw, [[13, 14.8], [16.3, 15]],
                       PSFConfig(sigma_x=1.3, sigma_y=1.3, noise_std=2.0))
    points = [[ion["x"], ion["y"]] for ion in result["ions"]]
    matched = match_points(points, truth, tolerance=.15)
    assert matched["all_correct"]
    assert matched["rmse_px"] < .1
    assert all(ion["status"] == "psf_fitted" for ion in result["ions"])


def test_fixed_motion_psf_merges_two_lobes_of_one_ion():
    shape, truth, motion, sigma = (33, 33), [16.1, 15.7], 3.2, .9
    raw = np.full(shape, 35.0)
    for phase in np.arange(16) * 2*np.pi/16:
        raw += gaussian_scene(shape, [(truth[0] + motion*np.cos(phase), truth[1], 700/16)],
                              sigma=sigma, background=0)
    result = fit_patch(raw, [[13.4, 15.7], [18.8, 15.7]],
                       PSFConfig(sigma_x=sigma, sigma_y=sigma, motion_amplitude=motion,
                                 noise_std=2.0, max_models=16, max_nfev=250))
    assert len(result["ions"]) == 1
    assert np.linalg.norm(np.array([result["ions"][0]["x"], result["ions"][0]["y"]]) - truth) < .15


def test_positive_residual_can_add_a_missing_neighbor():
    truth = [[12.3, 15.1], [16.4, 15.3]]
    raw = gaussian_scene((32, 32), [(*truth[0], 650), (*truth[1], 300)])
    result = fit_patch(raw, [[12.4, 15]], PSFConfig(sigma_x=1.3, sigma_y=1.3, noise_std=2.0))
    points = [[ion["x"], ion["y"]] for ion in result["ions"]]
    assert match_points(points, truth, tolerance=.2)["all_correct"]
    assert any(model["proposal"] == "positive_residual" for model in result["diagnostics"]["models"])


def test_model_selection_rejects_pure_background_candidate():
    result = fit_patch(np.full((24, 24), 100.0), [[12, 12]], PSFConfig(noise_std=2.0))
    assert result["ions"] == []
    assert result["diagnostics"]["selected_count"] == 0


def test_empty_candidates_and_empty_crystal_mask():
    raw = gaussian_scene((24, 24), [(12, 12, 500)])
    assert fit_patch(raw, [])["ions"] == []
    assert locate_ions(raw, region_mask=np.zeros(raw.shape, dtype=bool))["ions"] == []


def test_oversized_group_keeps_explicit_classical_fallbacks():
    raw = gaussian_scene((40, 40), [(12, 20, 600), (17, 20, 600), (22, 20, 600), (27, 20, 600)], sigma=.8)
    result = locate_ions(raw, PSFConfig(max_emitters=2, group_distance=20),
                         {"sigma": .3, "background_sigma": 8, "threshold_sigma": 3,
                          "min_area": 1, "min_distance": 2})
    assert len(result["ions"]) >= 3
    assert all(ion["status"] == "classical_fallback" for ion in result["ions"])
    assert result["diagnostics"]["fallback_ions"] == len(result["ions"])
    assert result["diagnostics"]["groups"][0]["status"] == "group_exceeds_max_emitters"


def test_failed_optimizer_does_not_report_successful_fit():
    raw = gaussian_scene((24, 24), [(12.4, 11.3, 600)])
    result = fit_patch(raw, [[10, 10]], PSFConfig(noise_std=2.0, max_nfev=1))
    assert result["diagnostics"]["status"] == "optimization_failed"
    assert result["ions"][0]["status"] == "classical_fallback"
    assert result["ions"][0]["x"] == 10


def test_render_does_not_renormalize_boundary_spots():
    cfg = PSFConfig(sigma_x=1.3, sigma_y=1.3)
    center = render_expected((32, 32), [{"x": 16, "y": 16, "amplitude_adu": 100}], config=cfg)
    edge = render_expected((32, 32), [{"x": 0, "y": 0, "amplitude_adu": 100}], config=cfg)
    assert edge.sum() < center.sum() * .6
    assert edge[0, 0] == pytest.approx(center[16, 16])


@pytest.mark.parametrize("options", [{"sigma_x": 0}, {"motion_amplitude": -1}, {"noise_std": 0},
                                    {"max_emitters": 0}, {"max_models": 1}, {"angle_deg": float("nan")}])
def test_invalid_psf_settings_fail(options):
    with pytest.raises(ValueError):
        PSFConfig(**options).validate()


def test_outside_region_seeds_are_rejected():
    allowed = np.zeros((24, 24), dtype=bool)
    allowed[10:15, 10:15] = True
    with pytest.raises(ValueError, match="inside region_mask"):
        fit_patch(np.ones((24, 24)), [[3, 3]], region_mask=allowed)


def test_patch_work_budget_is_enforced():
    with pytest.raises(ValueError, match="max_roi_pixels"):
        fit_patch(np.ones((70, 70)), [[30, 30]])


def test_cross_group_duplicate_favors_fitted_center():
    from ionrecog.psf_fitting.fitting import _deduplicate_groups
    rows = [{"x": 10.1, "y": 8.0, "fit_group": 0, "status": "classical_fallback"},
            {"x": 10.2, "y": 8.1, "fit_group": 1, "status": "psf_fitted"},
            {"x": 12.0, "y": 8.1, "fit_group": 2, "status": "psf_fitted"}]
    kept, removed = _deduplicate_groups(rows, .75)
    assert len(kept) == 2
    assert len(removed) == 1
    assert removed[0]["removed_group"] == 0
    assert any(row["fit_group"] == 1 and row["merged_fit_groups"] == [0, 1] for row in kept)


def test_cross_group_matching_does_not_chain_or_swallow_distinct_joint_points():
    from ionrecog.psf_fitting.fitting import _deduplicate_groups
    # The central competing hypothesis lies within tolerance of two distinct
    # jointly fitted points; it can represent only one of them, never both.
    rows = [{"x": 10.0, "y": 8.0, "fit_group": 0, "status": "psf_fitted"},
            {"x": 11.2, "y": 8.0, "fit_group": 0, "status": "psf_fitted"},
            {"x": 10.6, "y": 8.0, "fit_group": 1, "status": "psf_fitted", "fit_support_margin_px": 5}]
    kept, removed = _deduplicate_groups(rows, .75)
    assert len(kept) == 2
    assert len(removed) == 1
    # Strict boundary: points at or beyond the distance are untouched.
    separated = [{"x": 10.0, "y": 8.0, "fit_group": 0, "status": "psf_fitted"},
                 {"x": 10.75, "y": 8.0, "fit_group": 1, "status": "psf_fitted"},
                 {"x": 11.6, "y": 8.0, "fit_group": 2, "status": "psf_fitted"}]
    kept, removed = _deduplicate_groups(separated, .75)
    assert len(kept) == 3
    assert removed == []


def test_overlapping_fit_halos_reconcile_repeated_residual_emitter(monkeypatch):
    from ionrecog.psf_fitting import fitting
    # Candidate halos overlap, although neither initial candidate lies in the
    # other's halo. Both local fits discover the same previously missed point.
    raw = np.ones((30, 30))
    candidates = [{"x": 10.0, "y": 15.0}, {"x": 20.0, "y": 15.0}]
    monkeypatch.setattr(fitting, "detect_ions", lambda *a, **k: (candidates, np.ones(raw.shape, dtype=bool)))
    calls = []

    def local_result(image, seeds, config, region_mask=None):
        first = not calls
        calls.append(True)
        # With padding six, these ROIs start at x=4 and x=14.
        extra_x = 11.0 if first else 1.1
        return {"ions": [{"x": float(seeds[0, 0]), "y": float(seeds[0, 1]), "status": "psf_fitted"},
                         {"x": extra_x, "y": 6.0, "status": "psf_fitted"}],
                "diagnostics": {"rss": 10.0, "noise_std_adu": 2.0, "status": "fitted", "fallback": False}}

    monkeypatch.setattr(fitting, "fit_patch", local_result)
    result = locate_ions(raw, PSFConfig(sigma_x=1.0, sigma_y=1.0, padding=6))
    assert len(calls) == 2
    assert len(result["ions"]) == 3
    assert result["diagnostics"]["cross_group_duplicates_removed"] == 1
    assert len({ion["id"] for ion in result["ions"]}) == 3


def test_cli_registers_without_image_execution():
    import argparse
    from ionrecog.psf_fitting.cli import add_parser
    parser = argparse.ArgumentParser()
    add_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["psf", "--input", "example.png", "--output", "results", "--motion-amplitude", "2"])
    assert args.command == "psf"
    assert args.motion_amplitude == 2.0
    assert callable(args.handler)
