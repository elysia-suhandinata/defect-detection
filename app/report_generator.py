from typing import Any


def generate_inspection_report(result: dict[str, Any]) -> str:
    predictions = [
        prediction
        for prediction in result["predictions"]
        if prediction["detected"]
    ]

    if not predictions:
        return (
            "No defect was detected above the classification threshold. "
            "The steel sheet may proceed to the next inspection stage."
        )

    predictions.sort(
        key=lambda prediction: prediction["probability"],
        reverse=True,
    )

    primary = predictions[0]
    defect_class = primary["defect_class"]
    probability = primary["probability"]

    report = (
        f"A {defect_class} defect was detected with "
        f"{probability:.1%} confidence."
    )

    segmentation = result.get("segmentation", {})

    if segmentation.get("available"):
        location = segmentation.get("location")
        area = segmentation.get("defect_area_percent")

        if location:
            report += f" The defect is located near the {location} of the image."

        if area is not None:
            report += (
                f" The affected area covers approximately "
                f"{area:.1f}% of the steel sheet."
            )

    if probability >= 0.80:
        report += " Manual inspection and removal from the production line are recommended."
    elif probability >= 0.50:
        report += " Additional manual inspection is recommended."
    else:
        report += " The result should be reviewed before further action."

    return report