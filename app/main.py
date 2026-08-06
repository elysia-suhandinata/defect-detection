from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference import run_mock_inference
from app.report_generator import generate_inspection_report


app = FastAPI(
    title="Steel Defect Detection API",
    description=(
        "Detects steel defects and generates a plain-language "
        "inspection report."
    ),
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    allowed_types = {"image/jpeg", "image/png"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported.",
        )

    result = run_mock_inference(file.filename or "uploaded_image")
    result["inspection_report"] = await generate_inspection_report(result)

    return result