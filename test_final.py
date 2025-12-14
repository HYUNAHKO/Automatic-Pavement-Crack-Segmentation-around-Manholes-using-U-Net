import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import os
from tqdm import tqdm

MODEL_PATH = '/path/to/checkpoints/unet_final.pth'  # Update this path to your trained model checkpoint
INPUT_DIR = '/path/to/images'  # Update this path to your test images directory
OUTPUT_DIR = '/path/to/output/results_unet'  # Update this path to your desired output directory

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'overlays'), exist_ok=True)

device = 'cuda'

# Model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

checkpoint = torch.load(MODEL_PATH)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

print(f"Processing {INPUT_DIR}...")

image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')]

for img_file in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, img_file)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Preprocess
    resized = cv2.resize(image_rgb, (512, 512))
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    input_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Resize back
    pred_resized = cv2.resize(pred, (w, h))
    mask = (pred_resized > 0.5).astype(np.uint8) * 255
    
    # Save mask
    mask_path = os.path.join(OUTPUT_DIR, 'masks', img_file.replace('.jpg', '_mask.png'))
    cv2.imwrite(mask_path, mask)
    
    # Overlay
    overlay = image.copy()
    overlay[mask > 127] = [0, 0, 255]
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    overlay_path = os.path.join(OUTPUT_DIR, 'overlays', img_file)
    cv2.imwrite(overlay_path, blended)

print(f"Complete! Results in: {OUTPUT_DIR}")
