import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

from sources.features.segmentation.detector import (
    segment_pieces,
    extract_piece_images,
    save_results,
)

app = FastAPI(title="Jigsaw Puzzle Helper")

OUTPUT_DIR = Path("data/output")


@app.post("/api/segment")
async def segment(file: UploadFile):
    """Upload a photo and detect puzzle pieces. Returns count and saves results."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    result = segment_pieces(image)
    piece_images = extract_piece_images(image, result)

    filename = file.filename or "upload.jpg"
    saved = save_results(Path(filename), OUTPUT_DIR, result, piece_images)

    return {
        "detected_count": result.count,
        "bounding_boxes": result.bounding_boxes,
        "annotated_image": saved["annotated_image"],
        "pieces_dir": saved["pieces_dir"],
    }
