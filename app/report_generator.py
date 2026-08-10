import logging
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-5"

_SYSTEM_PROMPT = (
    "You are a quality-control assistant writing plain-language steel "
    "defect inspection reports for a production-line supervisor. You are "
    "given a classifier's per-class defect predictions and, when "
    "available, segmentation results for one steel sheet.\n\n"
    "Write a concise report (2-4 sentences, no headers or bullet points):\n"
    "- State plainly whether a defect was detected and, if so, which "
    "class.\n"
    "- If segmentation data is available, mention where the defect is and "
    "roughly how much of the sheet it covers.\n"
    "- End with one clear recommended action based on the primary "
    "defect's probability: at or above 0.80, recommend manual inspection "
    "and removal from the line; between 0.50 and 0.80, recommend "
    "additional manual inspection; below 0.50, recommend the result be "
    "reviewed before further action.\n"
    "- If no defect was detected, say so and state the sheet may proceed "
    "to the next inspection stage.\n\n"
    "Respond with only the report text."
)


async def generate_inspection_report(result: dict[str, Any]) -> str:
    try:
        return await _generate_llm_report(result)
    except Exception:
        logger.warning(
            "LLM report generation failed; falling back to template report.",
            exc_info=True,
        )
        return _generate_fallback_report(result)


async def _generate_llm_report(result: dict[str, Any]) -> str:
    client = anthropic.AsyncAnthropic()

    response = await client.messages.create(
        model=_MODEL,
        max_tokens=1024,
        output_config={"effort": "low"},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_prompt(result)}],
    )

    if response.stop_reason == "refusal":
        raise RuntimeError(f"Report generation refused: {response.stop_details}")

    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    if not text:
        raise RuntimeError("LLM returned an empty report")

    return text


def _build_prompt(result: dict[str, Any]) -> str:
    predictions = result.get("predictions", [])
    segmentation = result.get("segmentation", {})

    lines = [f"Filename: {result.get('filename', 'unknown')}", "", "Predictions:"]
    for prediction in predictions:
        lines.append(
            f"- {prediction['defect_class']}: "
            f"probability={prediction['probability']:.3f}, "
            f"detected={prediction['detected']}"
        )

    lines.append("")
    if segmentation.get("available"):
        area = segmentation.get("defect_area_percent")
        lines.append("Segmentation:")
        lines.append(f"- location: {segmentation.get('location', 'unknown')}")
        lines.append(
            f"- defect_area_percent: {area:.1f}"
            if area is not None
            else "- defect_area_percent: unknown"
        )
    else:
        lines.append("Segmentation: not available")

    return "\n".join(lines)


def _generate_fallback_report(result: dict[str, Any]) -> str:
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