"""Lazy CLI registration: --help does not construct or run any model."""


def _train(args):
    from .pipeline import train_model
    config = dict(patch_size=args.patch_size, max_shift=args.max_shift, channels=args.channels,
                  epochs=args.epochs, steps_per_epoch=args.steps, batch_size=args.batch_size,
                  views=args.views, val_steps=args.val_steps, learning_rate=args.lr,
                  seed=args.seed, noise_std=args.noise_std, num_threads=args.num_threads,
                  offset=args.offset, scale=args.scale)
    checkpoint = train_model(args.train, args.val, args.output, config,
                             device=args.device, confirm_single_ion=args.confirm_single_ion,
                             reserve_gib=args.reserve_gib)
    print("Checkpoint:", checkpoint)
    print("Validation measures geometric consistency, not localization accuracy.")


def _infer(args):
    from ..cli import load_config
    from ..localization_common import run_prediction
    from .pipeline import EquivariantRefiner, METHOD
    cfg = load_config(args.config)
    predictor = EquivariantRefiner(args.checkpoint, cfg["detection"], device=args.device,
                    max_refine_shift=args.max_refine_shift, isolation_distance=args.isolation_distance,
                    num_threads=args.num_threads)
    return run_prediction(args, predictor, METHOD, metadata={
        "checkpoint": str(args.checkpoint), "training_config": predictor.config,
        "max_refine_shift_px": predictor.max_refine_shift,
        "isolated_candidate_refinement_only": True,
        "official_lodestar_implementation": False,
        "note": "Classical proposals determine count. Crowded/edge/rejected refinements retain classical positions."})


def add_parser(subparsers):
    from ..localization_common import add_infer_arguments
    parser = subparsers.add_parser("equivariant", help="Geometric self-supervised isolated-ion candidate refinement")
    sub = parser.add_subparsers(dest="equivariant_command", required=True)
    train = sub.add_parser("train", help="Train on verified isolated single-ion crops, without center labels")
    train.set_defaults(handler=_train)
    train.add_argument("--train", required=True)
    train.add_argument("--val", required=True, help="Independent isolated crops, held out by source acquisition")
    train.add_argument("--output", required=True)
    train.add_argument("--confirm-single-ion", action="store_true", help="Assert every crop contains one full isolated ion; not a dense crystal")
    train.add_argument("--device", default="cpu")
    train.add_argument("--patch-size", type=int, default=17)
    train.add_argument("--max-shift", type=int, default=2)
    train.add_argument("--channels", type=int, default=16)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--steps", type=int, default=50, help="Minibatches per epoch")
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--views", type=int, default=3)
    train.add_argument("--val-steps", type=int, default=10)
    train.add_argument("--lr", type=float, default=.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--noise-std", type=float, default=.005, help="Extra independent augmentation noise, in normalized intensity")
    train.add_argument("--num-threads", type=int, default=2)
    train.add_argument("--offset", type=float, default=0.)
    train.add_argument("--scale", type=float, help="Fixed intensity scale; default fitted once from training maximum")
    train.add_argument("--reserve-gib", type=float, default=2.)
    infer = sub.add_parser("infer", help="Refine isolated classical proposals; report skipped crowded candidates")
    infer.set_defaults(handler=_infer)
    add_infer_arguments(infer)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--max-refine-shift", type=float, default=2.)
    infer.add_argument("--isolation-distance", type=float, help="Minimum distance to another proposal; conservative crop-based default")
    infer.add_argument("--num-threads", type=int, default=2)
    return parser
