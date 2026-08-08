from io import BytesIO
from pathlib import Path
from typing import Any
import sys

import torch
from PIL import Image
from torchvision import transforms

from app.models.classifier import DefectCNN


# Allow importing the U-Net implementation from src/rare_defect
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rare_defect.models import UNet


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


CLASS_NAMES = [
    "class_1",
    "class_2",
    "class_3",
    "class_4",
]

CLASSIFICATION_THRESHOLD = 0.5
SEGMENTATION_THRESHOLD = 0.5


# ---------------------------------------------------------
# Model paths
# ---------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "models"

CLASSIFIER_PATH = (
    MODEL_DIR / "gan_oversampled_cnn.pth"
)

UNET_PATH = (
    MODEL_DIR / "unet_weighted_best.pt"
)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

classifier_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

unet_transform = transforms.Compose(
    [
        transforms.Resize((256, 1600)),
        transforms.ToTensor(),
    ]
)


# ---------------------------------------------------------
# Load classifier
# ---------------------------------------------------------

classifier = DefectCNN().to(DEVICE)

classifier.load_state_dict(
    torch.load(
        CLASSIFIER_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

classifier.eval()


# ---------------------------------------------------------
# Load U-Net
# ---------------------------------------------------------

unet = UNet(
    in_channels=3,
    num_classes=4,
    base=32,
).to(DEVICE)

unet_checkpoint = torch.load(
    UNET_PATH,
    map_location=DEVICE,
    weights_only=True,
)

unet.load_state_dict(
    unet_checkpoint["model"]
)

unet.eval()


# ---------------------------------------------------------
# Classifier inference
# ---------------------------------------------------------

def run_classifier(
    image: Image.Image,
) -> list[dict[str, Any]]:

    image_tensor = (
        classifier_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        logits = classifier(image_tensor)

        probabilities = (
            torch.sigmoid(logits)[0]
            .cpu()
            .tolist()
        )

    predictions = []

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities,
    ):
        predictions.append(
            {
                "defect_class": class_name,
                "probability": float(probability),
                "detected": (
                    probability
                    >= CLASSIFICATION_THRESHOLD
                ),
            }
        )

    return predictions


# ---------------------------------------------------------
# Convert mask centroid into plain-language location
# ---------------------------------------------------------

def get_mask_location(
    mask: torch.Tensor,
) -> str | None:

    coordinates = torch.nonzero(mask)

    if coordinates.numel() == 0:
        return None

    center_y = (
        coordinates[:, 0]
        .float()
        .mean()
        .item()
        / mask.shape[0]
    )

    center_x = (
        coordinates[:, 1]
        .float()
        .mean()
        .item()
        / mask.shape[1]
    )

    if center_y < 1 / 3:
        vertical = "top"
    elif center_y < 2 / 3:
        vertical = "middle"
    else:
        vertical = "bottom"

    if center_x < 1 / 3:
        horizontal = "left"
    elif center_x < 2 / 3:
        horizontal = "center"
    else:
        horizontal = "right"

    if vertical == "middle" and horizontal == "center":
        return "center"

    if vertical == "middle":
        return horizontal

    if horizontal == "center":
        return vertical

    return f"{vertical}-{horizontal}"


# ---------------------------------------------------------
# U-Net inference
# ---------------------------------------------------------

def run_segmentation(
    image: Image.Image,
) -> dict[str, Any]:

    image_tensor = (
        unet_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        logits = unet(image_tensor)

        probabilities = torch.sigmoid(logits)[0]

        binary_masks = (
            probabilities
            > SEGMENTATION_THRESHOLD
        )

    binary_masks_cpu = binary_masks.cpu()

    # Combine all four defect-class masks
    combined_mask = binary_masks_cpu.any(dim=0)

    defect_area_percent = (
        combined_mask.float().mean().item()
        * 100
    )

    location = get_mask_location(
        combined_mask
    )

    per_class_area_percent = {}

    for class_name, mask in zip(
        CLASS_NAMES,
        binary_masks_cpu,
    ):
        area = (
            mask.float().mean().item()
            * 100
        )

        per_class_area_percent[class_name] = area

    return {
        "available": True,
        "location": location,
        "defect_area_percent": defect_area_percent,
        "per_class_area_percent": per_class_area_percent,
    }


# ---------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------

def run_inference(
    image_bytes: bytes,
    filename: str,
) -> dict[str, Any]:

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    predictions = run_classifier(
        image
    )

    segmentation = run_segmentation(
        image
    )

    return {
        "filename": filename,
        "predictions": predictions,
        "segmentation": segmentation,
    }