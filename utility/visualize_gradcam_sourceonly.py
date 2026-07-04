import argparse
import os
import random
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torchvision import models, transforms


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw Grad-CAM maps for a source-only ResNet on real target-domain images."
    )
    parser.add_argument("--rootpath", default=".", help="Project root path.")
    parser.add_argument("--source", default="SD", choices=["SD", "Road", "DECKGPRH"])
    parser.add_argument("--target", default="Road", choices=["SD", "Road", "DECKGPRH"])
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet101"])
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to source-domain CNN checkpoint. If omitted, use Feature-Generation-datasets/<source>/res50_finetuned.pth or res101_finetuned.pth.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Single image path. If omitted, images are sampled from data/<target>/filename_test.txt or img folder.",
    )
    parser.add_argument(
        "--file_list",
        default=None,
        help="Optional filename list under data/<target>. Default: data/<target>/filename_test.txt.",
    )
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--samples_per_class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay opacity.")
    parser.add_argument(
        "--style",
        default="paper",
        choices=["paper", "analysis", "triplet"],
        help="paper saves a clean Grad-CAM overlay like common papers; analysis adds contours; triplet saves image/CAM/overlay.",
    )
    parser.add_argument(
        "--target_layer",
        default="layer4",
        choices=["layer2", "layer3", "layer4"],
        help="ResNet layer used for Grad-CAM. layer3 is usually clearer for subtle GPR/B-scan images.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.8,
        help="Contrast enhancement for visualization only. Model input is unchanged.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.75,
        help="Gamma applied to the normalized CAM for visualization. Lower values reveal weaker regions.",
    )
    parser.add_argument(
        "--cam_target",
        default="pred",
        choices=["pred", "true"],
        help="Use predicted source class or true filename class as the Grad-CAM target.",
    )
    parser.add_argument(
        "--target_class",
        default=None,
        help="Force CAM target by source class name or numeric class index, e.g. crack or 2.",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--outdir", default="outputs/gradcam_sourceonly")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_names_from_images(rootpath, dataset):
    img_dir = os.path.join(rootpath, "data", dataset, "img")
    names = set()
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            names.add(filename.split("-")[0])
    return sorted(names)


def default_checkpoint(rootpath, source, backbone):
    filename = "res50_finetuned.pth" if backbone == "resnet50" else "res101_finetuned.pth"
    return os.path.join(rootpath, "Feature-Generation-datasets", source, filename)


def build_model(backbone, num_classes):
    if backbone == "resnet50":
        model = models.resnet50(weights=None)
    else:
        model = models.resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_state_dict(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Missing checkpoint keys: {missing[:8]}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected[:8]}")


def preprocess_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def read_file_list(rootpath, target, file_list):
    if file_list is None:
        file_list = os.path.join(rootpath, "data", target, "filename_test.txt")
    elif not os.path.isabs(file_list):
        file_list = os.path.join(rootpath, file_list)

    if not os.path.exists(file_list):
        return None

    with open(file_list, "r", encoding="utf-8") as handle:
        names = [line.strip() for line in handle if line.strip()]
    return names


def resolve_image_path(rootpath, target, name):
    if os.path.isabs(name) and os.path.exists(name):
        return name

    img_dir = os.path.join(rootpath, "data", target, "img")
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    candidates = [base] if ext else [stem + suffix for suffix in IMAGE_EXTENSIONS]
    for candidate in candidates:
        path = os.path.join(img_dir, candidate)
        if os.path.exists(path):
            return path
    return None


def sample_target_images(rootpath, target, file_list, num_images, samples_per_class, seed):
    if file_list is not None:
        names = read_file_list(rootpath, target, file_list)
    else:
        names = read_file_list(rootpath, target, None)

    if names is None:
        img_dir = os.path.join(rootpath, "data", target, "img")
        names = sorted(
            filename
            for filename in os.listdir(img_dir)
            if filename.lower().endswith(IMAGE_EXTENSIONS)
        )

    paths = []
    for name in names:
        path = resolve_image_path(rootpath, target, name)
        if path is not None:
            paths.append(path)

    grouped = {}
    for path in paths:
        label_name = os.path.basename(path).split("-")[0]
        grouped.setdefault(label_name, []).append(path)

    rng = random.Random(seed)
    selected = []
    for label_name in sorted(grouped):
        group = grouped[label_name][:]
        rng.shuffle(group)
        selected.extend(group[:samples_per_class])

    if len(selected) < num_images:
        remaining = [path for path in paths if path not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: num_images - len(selected)])

    return selected[:num_images]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        pred_idx = int(logits.argmax(dim=1).item())
        if class_idx is None:
            class_idx = pred_idx

        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        cam_map = F.relu(cam_map)
        cam_map = F.interpolate(
            cam_map,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_map = cam_map.squeeze().detach().cpu().numpy()
        cam_map -= cam_map.min()
        cam_map /= cam_map.max() + 1e-8
        probs = F.softmax(logits.detach(), dim=1).squeeze().cpu().numpy()
        return cam_map, pred_idx, probs


def colorize_cam(cam_map):
    colored = cm.get_cmap("jet")(cam_map)[..., :3]
    return (colored * 255).astype(np.uint8)


def enhance_for_display(image, contrast):
    image = image.convert("RGB")
    image = ImageOps.autocontrast(image)
    if contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


def make_overlay(image, cam_map, alpha):
    image = image.convert("RGB")
    heatmap = Image.fromarray(colorize_cam(cam_map)).resize(image.size, Image.BILINEAR)
    return Image.blend(image, heatmap, alpha=alpha)


def parse_forced_class(target_class, source_classes):
    if target_class is None:
        return None
    if target_class.isdigit():
        return int(target_class)
    if target_class not in source_classes:
        raise ValueError(f"Unknown target_class '{target_class}'. Source classes: {source_classes}")
    return source_classes.index(target_class)


def true_class_index_from_filename(path, source_classes):
    class_name = os.path.basename(path).split("-")[0]
    if class_name not in source_classes:
        return None
    return source_classes.index(class_name)


def save_triplet(original, cam_map, overlay, output_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    axes[0].imshow(original)
    axes[0].set_title("Image")
    axes[1].imshow(cam_map, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_paper_overlay(overlay, output_path):
    overlay.save(output_path)


def save_raw_heatmap(cam_map, output_path):
    Image.fromarray(colorize_cam(cam_map)).save(output_path)


def save_overlay(original, cam_map, overlay, output_path, title):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.imshow(overlay)
    height, width = cam_map.shape
    xs = np.linspace(0, original.size[0], width)
    ys = np.linspace(0, original.size[1], height)
    grid_x, grid_y = np.meshgrid(xs, ys)
    ax.contour(
        grid_x,
        grid_y,
        cam_map,
        levels=[0.45, 0.65, 0.82],
        colors=["white", "yellow", "red"],
        linewidths=[0.8, 1.0, 1.2],
        alpha=0.9,
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout(pad=0.15)
    fig.savefig(output_path, dpi=400, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def get_target_layer(model, layer_name):
    if layer_name == "layer2":
        return model.layer2[-1]
    if layer_name == "layer3":
        return model.layer3[-1]
    return model.layer4[-1]


def main():
    args = parse_args()
    args.rootpath = os.path.abspath(args.rootpath)
    args.outdir = os.path.abspath(os.path.join(args.rootpath, args.outdir))
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    source_classes = class_names_from_images(args.rootpath, args.source)
    checkpoint_path = args.checkpoint or default_checkpoint(args.rootpath, args.source, args.backbone)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(args.backbone, num_classes=len(source_classes))
    load_state_dict(model, checkpoint_path, device)
    model.to(device)
    model.eval()

    target_layer = get_target_layer(model, args.target_layer)
    gradcam = GradCAM(model, target_layer)
    transform = preprocess_transform(args.image_size)
    forced_class = parse_forced_class(args.target_class, source_classes)

    if args.image is not None:
        image_paths = [resolve_image_path(args.rootpath, args.target, args.image)]
        image_paths = [path for path in image_paths if path is not None]
    else:
        image_paths = sample_target_images(
            args.rootpath,
            args.target,
            args.file_list,
            args.num_images,
            args.samples_per_class,
            args.seed,
        )

    print(f"Source classes: {source_classes}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Images: {len(image_paths)}")

    summary_path = os.path.join(
        args.outdir, f"{args.source}_to_{args.target}_{args.backbone}_gradcam_summary.csv"
    )
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("image,true_class,cam_class,pred_class,pred_prob,output\n")
        for path in image_paths:
            original = Image.open(path).convert("RGB")
            input_tensor = transform(original).unsqueeze(0).to(device)

            cam_class = forced_class
            if cam_class is None and args.cam_target == "true":
                cam_class = true_class_index_from_filename(path, source_classes)

            cam_map, pred_idx, probs = gradcam(input_tensor, class_idx=cam_class)
            cam_map = np.power(cam_map, args.gamma)
            if cam_class is None:
                cam_class = pred_idx

            resized_original = original.resize((args.image_size, args.image_size), Image.BILINEAR)
            display_original = enhance_for_display(resized_original, args.contrast)
            overlay = make_overlay(display_original, cam_map, args.alpha)
            stem = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(args.outdir, f"{stem}_gradcam.png")
            heatmap_path = os.path.join(args.outdir, f"{stem}_heatmap.png")
            title = (
                f"{args.source}->{args.target} | "
                f"CAM: {source_classes[cam_class]} | "
                f"Pred: {source_classes[pred_idx]} ({probs[pred_idx]:.2f})"
            )
            if args.style == "paper":
                save_paper_overlay(overlay, output_path)
                # save_raw_heatmap(cam_map, heatmap_path)
            elif args.style == "analysis":
                save_overlay(display_original, cam_map, overlay, output_path, title)
                # save_raw_heatmap(cam_map, heatmap_path)
            else:
                save_triplet(display_original, cam_map, overlay, output_path, title)
            true_class = os.path.basename(path).split("-")[0]
            handle.write(
                f"{path},{true_class},{source_classes[cam_class]},"
                f"{source_classes[pred_idx]},{probs[pred_idx]:.6f},{output_path}\n"
            )
            print(f"Saved: {output_path}")

    gradcam.remove_hooks()
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
