"""Independent point-localization commands; parser construction does not load torch."""


def _run_train(args):
    from .simulation import read_simulation_config
    from .training import train_model
    simulation = read_simulation_config(args.simulation)
    if args.size is not None:
        simulation.size = args.size
        simulation.validate()
    result = train_model(args.output, simulation, args.epochs, args.steps, args.batch_size,
                         args.val_samples, args.channels, args.lr, args.seed, args.device,
                         args.num_threads, args.reserve_gib, args.resume,
                         protected_sources=[args.simulation] if args.simulation else [])
    print("Point checkpoint:", result["best_checkpoint"])
    print("Validation describes simulation only; no experimental localization accuracy has been established.")


def _run_infer(args):
    from ..localization_common import run_prediction
    from .inference import PointPredictor
    predictor = PointPredictor(args.checkpoint, args.device, args.threshold, args.min_distance,
                               args.max_ions, args.tile_size, args.halo, args.support_snr, args.num_threads)
    return run_prediction(args, predictor, "point_localization", metadata={
        "checkpoint": str(predictor.checkpoint.resolve()), "training_source": "simulation_supervised",
        "official_DECODE_or_LiteLoc_reproduction": False})


def populate_parser(parser):
    from ..localization_common import add_infer_arguments
    sub = parser.add_subparsers(dest="point_action", required=True)
    train = sub.add_parser("train", help="Train center probability and subpixel offsets using online simulation")
    train.add_argument("--output", required=True)
    train.add_argument("--simulation", help="JSON SimulatorConfig; controls optical/noise/normalization assumptions")
    train.add_argument("--size", type=int, help="Override simulation crop size")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--steps", type=int, default=100, help="Optimization steps per epoch")
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--val-samples", type=int, default=16)
    train.add_argument("--channels", type=int, default=16)
    train.add_argument("--lr", type=float, default=.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    train.add_argument("--num-threads", type=int, default=2)
    train.add_argument("--reserve-gib", type=float, default=2.)
    train.add_argument("--resume", help="Trusted last.pt; keep original settings, new output, increase total epochs")
    train.set_defaults(handler=_run_train)
    infer = sub.add_parser("infer", help="Predict centers directly in raw images")
    add_infer_arguments(infer)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    infer.add_argument("--num-threads", type=int, default=2)
    infer.add_argument("--threshold", type=float, help="Override threshold selected on synthetic validation")
    infer.add_argument("--min-distance", type=float, default=1.5)
    infer.add_argument("--max-ions", type=int, default=5000)
    infer.add_argument("--tile-size", type=int, default=256)
    infer.add_argument("--halo", type=int, default=32)
    infer.add_argument("--support-snr", type=float, default=0., help="Optional raw-signal support threshold; 0 disables it")
    infer.set_defaults(handler=_run_infer)


def add_parser(subparsers):
    parser = subparsers.add_parser("points", help="Simulation-supervised direct point localization (experimental)")
    populate_parser(parser)
    return parser


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Direct point localization research baseline")
    populate_parser(parser)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, "Error: %s\n" % exc)
