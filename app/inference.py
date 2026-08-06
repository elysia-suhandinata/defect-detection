from typing import Any


def run_mock_inference(filename: str) -> dict[str, Any]:
    return {
        "filename": filename,
        "predictions": [
            {
                "defect_class": "class_1",
                "probability": 0.12,
                "detected": False,
            },
            {
                "defect_class": "class_2",
                "probability": 0.18,
                "detected": False,
            },
            {
                "defect_class": "class_3",
                "probability": 0.87,
                "detected": True,
            },
            {
                "defect_class": "class_4",
                "probability": 0.09,
                "detected": False,
            },
        ],
        "segmentation": {
            "available": False,
            "location": None,
            "defect_area_percent": None,
        },
    }