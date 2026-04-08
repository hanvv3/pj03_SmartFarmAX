from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


def normalize_label_to_name(raw_mapping: dict | None) -> dict[int, str]:
    raw_mapping = raw_mapping or {}
    return {int(label): str(name) for label, name in raw_mapping.items()}


def normalize_disease_to_label(raw_mapping: dict | None) -> dict[int, int]:
    raw_mapping = raw_mapping or {}
    return {int(disease): int(label) for disease, label in raw_mapping.items()}


def infer_convnext_fpn_config(state_dict: dict) -> tuple[dict[str, str], list[int]]:
    body_keys = tuple(state_dict.keys())
    has_stage2 = any(key.startswith("backbone.body.2.") for key in body_keys)
    has_stage4 = any(key.startswith("backbone.body.4.") for key in body_keys)
    has_stage6 = any(key.startswith("backbone.body.6.") for key in body_keys)
    has_stage7 = any(key.startswith("backbone.body.7.") for key in body_keys)

    # Older strawberry checkpoint layout.
    if has_stage2 and has_stage4 and has_stage6 and not has_stage7:
        return {"2": "0", "4": "1", "6": "2"}, [192, 384, 768]

    # Newer lettuce-style layout.
    if has_stage7:
        return {"3": "0", "5": "1", "7": "2"}, [192, 384, 768]

    # Fallback for checkpoints that end at stage 6 but use the newer layout.
    if has_stage6:
        return {"3": "0", "5": "1", "6": "2"}, [192, 384, 768]

    raise ValueError("Unable to infer ConvNeXt FPN layout from checkpoint state_dict.")


def build_resnet50_fasterrcnn(num_classes: int, image_min_size: int, image_max_size: int) -> FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        min_size=image_min_size,
        max_size=image_max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_convnext_fasterrcnn(
    num_classes: int,
    image_min_size: int,
    image_max_size: int,
    variant: str = "small",
    return_layers: dict[str, str] | None = None,
    in_channels_list: list[int] | None = None,
) -> FasterRCNN:
    from torchvision.models import convnext_small, convnext_tiny
    from torchvision.models.detection.backbone_utils import BackboneWithFPN

    builders = {
        "tiny": lambda: convnext_tiny(weights=None).features,
        "small": lambda: convnext_small(weights=None).features,
    }
    if variant not in builders:
        raise ValueError(f"Unsupported ConvNeXt variant: {variant}")

    backbone_body = builders[variant]()
    backbone = BackboneWithFPN(
        backbone=backbone_body,
        return_layers=return_layers or {"3": "0", "5": "1", "7": "2"},
        in_channels_list=in_channels_list or [192, 384, 768],
        out_channels=256,
    )
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "pool"],
        output_size=7,
        sampling_ratio=2,
    )
    return FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=image_min_size,
        max_size=image_max_size,
    )


def build_model_from_checkpoint(
    checkpoint: dict,
    state_dict: dict | None = None,
    image_min_size: int = 640,
    image_max_size: int = 640,
) -> FasterRCNN:
    state_dict = state_dict or checkpoint["model_state_dict"]
    model_config = checkpoint.get("model_config", {}) or {}
    best_params = checkpoint.get("best_params", {}) or {}

    backbone_name = model_config.get("backbone") or best_params.get("backbone", "convnext_small")
    num_classes = int(checkpoint.get("num_classes", len(checkpoint.get("class_names", {})) + 1))
    image_min_size = int(model_config.get("image_min_size", image_min_size))
    image_max_size = int(model_config.get("image_max_size", image_max_size))

    if backbone_name == "resnet50":
        return build_resnet50_fasterrcnn(num_classes, image_min_size, image_max_size)

    if backbone_name in ("convnext_tiny", "convnext_small"):
        return_layers = model_config.get("fpn_return_layers")
        in_channels_list = model_config.get("in_channels_list")
        if return_layers is None or in_channels_list is None:
            return_layers, in_channels_list = infer_convnext_fpn_config(state_dict)
        return build_convnext_fasterrcnn(
            num_classes=num_classes,
            image_min_size=image_min_size,
            image_max_size=image_max_size,
            variant=backbone_name.split("_")[1],
            return_layers={str(k): str(v) for k, v in return_layers.items()},
            in_channels_list=[int(v) for v in in_channels_list],
        )

    raise ValueError(f"Unsupported backbone: {backbone_name}")


def load_detection_checkpoint(
    model_path: str | Path,
    device: torch.device,
    image_min_size: int = 640,
    image_max_size: int = 640,
):
    model_path = Path(model_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    model = build_model_from_checkpoint(
        checkpoint,
        state_dict=state_dict,
        image_min_size=image_min_size,
        image_max_size=image_max_size,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    best_params = checkpoint.get("best_params", {}) or {}
    backbone_name = checkpoint.get("model_config", {}).get("backbone") or best_params.get("backbone", "convnext_small")
    label_to_name = normalize_label_to_name(checkpoint.get("class_names"))
    disease_to_label = normalize_disease_to_label(checkpoint.get("disease_to_label"))
    num_classes = int(checkpoint.get("num_classes", len(label_to_name) + 1))

    model_config = checkpoint.get("model_config", {}) or {}
    fpn_return_layers = model_config.get("fpn_return_layers")
    if fpn_return_layers is None:
        fpn_return_layers, _ = infer_convnext_fpn_config(state_dict) if backbone_name.startswith("convnext_") else ({}, [])

    return {
        "model": model,
        "checkpoint": checkpoint,
        "state_dict": state_dict,
        "backbone_name": backbone_name,
        "label_to_name": label_to_name,
        "disease_to_label": disease_to_label,
        "num_classes": num_classes,
        "fpn_return_layers": fpn_return_layers,
    }
