"""Template inspection report (optional Module 8 thin path).

Severstal ClassIds are unnamed; glossary below is project-local only.
"""

from __future__ import annotations

from dataclasses import dataclass

CLASS_GLOSSARY = {
    1: "surface type-1 defect",
    2: "surface type-2 defect (rare)",
    3: "surface type-3 defect",
    4: "surface type-4 defect",
}


@dataclass
class DetectionSummary:
    image_id: str
    class_probs: dict[int, float]
    mask_areas: dict[int, float]
    recommended_action: str


def choose_action(summary: DetectionSummary, fn_cost: float, fp_cost: float) -> str:
    if summary.mask_areas.get(2, 0.0) > 0 or summary.class_probs.get(2, 0.0) > 0.5:
        return "HOLD — rare defect signal; route to human review"
    if any(a > 0 for a in summary.mask_areas.values()):
        return "FLAG — defect present; secondary inspection"
    if max(summary.class_probs.values(), default=0) > 0.2 and fn_cost >= 5 * fp_cost:
        return "SOFT-FLAG — low confidence; sample for audit"
    return "PASS — no defect above threshold"


def render_template_report(summary: DetectionSummary) -> str:
    lines = [f"Inspection report for {summary.image_id}", "Detected classes:"]
    for cid in sorted(summary.class_probs):
        name = CLASS_GLOSSARY.get(cid, f"class {cid}")
        lines.append(
            f"  - {name}: confidence={summary.class_probs[cid]:.2f}, "
            f"mask_area={summary.mask_areas.get(cid, 0.0):.4%}"
        )
    if not summary.class_probs:
        lines.append("  - none")
    lines.append(f"Recommended action: {summary.recommended_action}")
    lines.append("Note: Class names are placeholders; Severstal publishes numeric ClassIds only.")
    return "\n".join(lines)


def build_report_from_masks(image_id, probs, areas, fn_cost=10.0, fp_cost=1.0) -> str:
    summary = DetectionSummary(image_id, probs, areas, "")
    summary.recommended_action = choose_action(summary, fn_cost, fp_cost)
    return render_template_report(summary)
