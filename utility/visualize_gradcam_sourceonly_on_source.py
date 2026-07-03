import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convenience wrapper: draw Grad-CAM for a source-only CNN on its own source-domain images."
    )
    parser.add_argument("--rootpath", default=".", help="Project root path.")
    parser.add_argument("--source", default="SD", choices=["SD", "Road", "DECKGPRH"])
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument("--checkpoint", default=None, help="Optional source-only CNN checkpoint path.")
    parser.add_argument("--image", default=None, help="Optional single source image path or image name.")
    parser.add_argument("--file_list", default=None, help="Optional source-domain filename list.")
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--samples_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--target_layer", default="layer4", choices=["layer2", "layer3", "layer4"])
    parser.add_argument("--contrast", type=float, default=1.8)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--style", default="paper", choices=["paper", "analysis", "triplet"])
    parser.add_argument("--cam_target", default="pred", choices=["pred", "true"])
    parser.add_argument("--target_class", default=None)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--outdir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    script_path = os.path.join(os.path.dirname(__file__), "visualize_gradcam_sourceonly.py")
    outdir = args.outdir or os.path.join("outputs", "gradcam_sourceonly_on_source", args.source)

    command = [
        sys.executable,
        script_path,
        "--rootpath",
        args.rootpath,
        "--source",
        args.source,
        "--target",
        args.source,
        "--backbone",
        args.backbone,
        "--num_images",
        str(args.num_images),
        "--samples_per_class",
        str(args.samples_per_class),
        "--seed",
        str(args.seed),
        "--image_size",
        str(args.image_size),
        "--alpha",
        str(args.alpha),
        "--target_layer",
        args.target_layer,
        "--contrast",
        str(args.contrast),
        "--gamma",
        str(args.gamma),
        "--style",
        args.style,
        "--cam_target",
        args.cam_target,
        "--outdir",
        outdir,
    ]

    if args.checkpoint is not None:
        command.extend(["--checkpoint", args.checkpoint])
    if args.image is not None:
        command.extend(["--image", args.image])
    if args.file_list is not None:
        command.extend(["--file_list", args.file_list])
    if args.target_class is not None:
        command.extend(["--target_class", args.target_class])
    if args.cuda:
        command.append("--cuda")

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
