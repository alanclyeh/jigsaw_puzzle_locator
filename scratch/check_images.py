import cv2
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
data_dir = project_root / "data"

ref_path = data_dir / "reference_puzzle.jpg"
if ref_path.exists():
    img = cv2.imread(str(ref_path))
    if img is not None:
        print(f"大圖 reference_puzzle.jpg 尺寸: {img.shape}")
        
for p_file in data_dir.glob("*.jpg"):
    if p_file.name != "reference_puzzle.jpg":
        img = cv2.imread(str(p_file))
        if img is not None:
            print(f"單片 {p_file.name} 尺寸: {img.shape}")
