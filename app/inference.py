from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from app.models.classifier import DefectCNN


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

CLASS_NAMES = [
    "class_1",
    "class_2",
    "class_3",
    "class_4",
]

THRESHOLD = 0.5

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "gan_oversampled_cnn.pth"
)

transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)


classifier = DefectCNN().to(DEVICE)

classifier.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )
)

classifier.eval()


def run_inference(
    image_bytes: bytes,
    filename: str,
) -> dict[str, Any]:

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    image_tensor = (
        transform(image)
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
                "detected": probability >= THRESHOLD,
            }
        )

    return {
        "filename": filename,
        "predictions": predictions,

        # U-Net will be connected next
        "segmentation": {
            "available": False,
            "location": None,
            "defect_area_percent": None,
        },
    }