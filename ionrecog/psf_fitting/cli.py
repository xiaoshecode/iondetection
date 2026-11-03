"""PSF command registration. No image analysis occurs during import/help."""
from dataclasses import asdict


def add_parser(subparsers):
    from ..localization_common import add_infer_arguments
    parser = subparsers.add_parser("psf", help="Joint fixed-PSF localization with explicit classical fallbacks")
    add_infer_arguments(parser)
    parser.add_argument("--sigma-x", type=float, default=1.4, help="Fixed optical Gaussian sigma, pixels")
    parser.add_argument("--sigma-y", type=float, default=1.4)
    parser.add_argument("--psf-angle", type=float, default=0.0, help="Optical ellipse angle, degrees")
    parser.add_argument("--motion-amplitude", type=float, default=0.0, help="Known symmetric sinusoidal motion amplitude, pixels")
    parser.add_argument("--motion-angle", type=float, default=0.0, help="Motion direction, degrees")
    parser.add_argument("--motion-samples", type=int, default=16)
    parser.add_argument("--noise-std", type=float, help="Constant residual noise std in raw ADU; otherwise robust estimate")
    parser.add_argument("--group-distance", type=float, default=6.0)
    parser.add_argument("--padding", type=int, default=6)
    parser.add_argument("--max-emitters", type=int, default=5)
    parser.add_argument("--max-roi-pixels", type=int, default=4096)
    parser.add_argument("--max-fit-groups", type=int, default=256)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--max-models", type=int, default=12)
    parser.add_argument("--selection-rounds", type=int, default=3)
    parser.add_argument("--min-delta-score", type=float, default=6.0)
    parser.add_argument("--residual-sigma", type=float, default=4.0)
    parser.add_argument("--min-separation", type=float, default=0.75)
    parser.set_defaults(handler=run)
    return parser


def run(args):
    from ..cli import load_config
    from ..localization_common import run_prediction
    from .fitting import PSFConfig, locate_ions
    base = load_config(args.config)
    cfg = PSFConfig(sigma_x=args.sigma_x, sigma_y=args.sigma_y, angle_deg=args.psf_angle,
                    motion_amplitude=args.motion_amplitude, motion_angle_deg=args.motion_angle,
                    motion_samples=args.motion_samples, noise_std=args.noise_std,
                    group_distance=args.group_distance, padding=args.padding, max_emitters=args.max_emitters,
                    max_roi_pixels=args.max_roi_pixels, max_fit_groups=args.max_fit_groups,
                    max_nfev=args.max_nfev, max_models=args.max_models, selection_rounds=args.selection_rounds,
                    min_delta_score=args.min_delta_score, residual_sigma=args.residual_sigma,
                    min_separation=args.min_separation, radius=float(base["detection"]["radius"])).validate()
    return run_prediction(args, lambda raw, region_mask: locate_ions(raw, cfg, base["detection"], region_mask),
                          "psf", metadata={"psf_config": asdict(cfg), "training_required": False,
                                           "implementation": "fixed analytical PSF baseline, not a paper reproduction"})
