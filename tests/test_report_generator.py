import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.report_generator import _generate_fallback_report, generate_inspection_report


def _mock_response(text: str, stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=None,
        content=[SimpleNamespace(type="text", text=text)],
    )


class GenerateInspectionReportTests(unittest.IsolatedAsyncioTestCase):
    def _result(self, **overrides):
        result = {
            "filename": "sample.jpg",
            "predictions": [
                {"defect_class": "class_1", "probability": 0.12, "detected": False},
                {"defect_class": "class_2", "probability": 0.18, "detected": False},
                {"defect_class": "class_3", "probability": 0.87, "detected": True},
                {"defect_class": "class_4", "probability": 0.09, "detected": False},
            ],
            "segmentation": {
                "available": False,
                "location": None,
                "defect_area_percent": None,
            },
        }
        result.update(overrides)
        return result

    @patch("app.report_generator.anthropic.AsyncAnthropic")
    async def test_uses_llm_report_when_available(self, mock_anthropic_cls):
        mock_client = mock_anthropic_cls.return_value
        mock_client.messages.create = AsyncMock(
            return_value=_mock_response(
                "A class_3 defect was found; pull the sheet for inspection."
            )
        )

        report = await generate_inspection_report(self._result())

        self.assertEqual(
            report, "A class_3 defect was found; pull the sheet for inspection."
        )
        mock_client.messages.create.assert_called_once()
        _, kwargs = mock_client.messages.create.call_args
        self.assertEqual(kwargs["model"], "claude-opus-5")
        self.assertIn("class_3", kwargs["messages"][0]["content"])

    @patch("app.report_generator.anthropic.AsyncAnthropic")
    async def test_falls_back_to_template_when_llm_call_fails(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create = AsyncMock(
            side_effect=RuntimeError("no api key configured")
        )

        report = await generate_inspection_report(self._result())

        self.assertIn("class_3", report)
        self.assertIn("87.0%", report)
        self.assertIn("removal from the production line", report)

    @patch("app.report_generator.anthropic.AsyncAnthropic")
    async def test_falls_back_when_llm_refuses(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create = AsyncMock(
            return_value=_mock_response("", stop_reason="refusal")
        )

        report = await generate_inspection_report(self._result())

        self.assertIn("class_3", report)

    @patch("app.report_generator.anthropic.AsyncAnthropic")
    async def test_falls_back_when_llm_returns_empty_text(self, mock_anthropic_cls):
        mock_anthropic_cls.return_value.messages.create = AsyncMock(
            return_value=_mock_response("   ")
        )

        report = await generate_inspection_report(self._result())

        self.assertIn("class_3", report)


class FallbackReportTests(unittest.TestCase):
    def _result(self, **overrides):
        result = {
            "filename": "sample.jpg",
            "predictions": [
                {"defect_class": "class_1", "probability": 0.05, "detected": False},
            ],
            "segmentation": {
                "available": False,
                "location": None,
                "defect_area_percent": None,
            },
        }
        result.update(overrides)
        return result

    def test_no_defect_detected(self):
        report = _generate_fallback_report(self._result())

        self.assertEqual(
            report,
            "No defect was detected above the classification threshold. "
            "The steel sheet may proceed to the next inspection stage.",
        )

    def test_high_confidence_recommends_removal(self):
        result = self._result(
            predictions=[
                {"defect_class": "class_3", "probability": 0.87, "detected": True},
            ]
        )

        report = _generate_fallback_report(result)

        self.assertIn("class_3", report)
        self.assertIn("87.0%", report)
        self.assertIn("removal from the production line", report)

    def test_low_confidence_recommends_review(self):
        result = self._result(
            predictions=[
                {"defect_class": "class_2", "probability": 0.35, "detected": True},
            ]
        )

        report = _generate_fallback_report(result)

        self.assertIn("reviewed before further action", report)

    def test_includes_segmentation_details_when_available(self):
        result = self._result(
            predictions=[
                {"defect_class": "class_2", "probability": 0.6, "detected": True},
            ],
            segmentation={
                "available": True,
                "location": "top-left corner",
                "defect_area_percent": 4.2,
            },
        )

        report = _generate_fallback_report(result)

        self.assertIn("top-left corner", report)
        self.assertIn("4.2%", report)
        self.assertIn("Additional manual inspection", report)


if __name__ == "__main__":
    unittest.main()
