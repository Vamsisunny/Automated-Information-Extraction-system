import os
import sys

# 1. Essential Windows Fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_SIGNAL_HANDLER"] = "0" # Prevents silent exit on signal errors

print("--- Step 1: Importing Paddle ---", flush=True)
try:
    from paddleocr import PaddleOCR
    print("✅ Paddle Imported!", flush=True)
except Exception as e:
    print(f"❌ Import failed: {e}", flush=True)
    sys.exit()

print("--- Step 2: Initializing Engine (CPU Mode) ---", flush=True)
try:
    # use_gpu=False is crucial here to prevent the silent driver crash
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=True)
    print("✅ Engine Created!", flush=True)
except Exception as e:
    print(f"❌ Creation failed: {e}", flush=True)
    sys.exit()

# Use the full absolute path to be 100% sure
base_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(base_dir, "data", "archive", "SROIE2019", "train", "img", "000.jpg")

print(f"--- Step 3: Checking path: {img_path} ---", flush=True)
if os.path.exists(img_path):
    print("✅ Image found! Running OCR...", flush=True)
    # This is where the crash usually happens
    result = ocr.ocr(img_path, cls=True)
    print("\n--- OCR SUCCESS! ---", flush=True)
    print(result, flush=True)
else:
    print("❌ Image file not found at that location!", flush=True)

print("--- End of Script ---", flush=True)