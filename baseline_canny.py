import cv2
import numpy as np
import os
from tqdm import tqdm
import json

INPUT_DIR = '/path/to/images'  # Update this path to your images directory
OUTPUT_DIR = '/path/to/output/canny_baseline'  # Update this path to your output directory
GT_MASK_DIR = '/path/to/ground_truth_masks'  # Update this path to your ground truth masks directory

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'masks'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'overlays'), exist_ok=True)


def canny_baseline(image_path):
    """
    Naive Baseline: Canny Edge Detection
    
    Expected to fail on:
    - Shadow boundaries
    - Manhole rim patterns
    - Asphalt texture noise
    """
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological closing to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return closed, image


def calculate_iou(pred_mask, gt_mask):
    """Calculate IoU"""
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_dice(pred_mask, gt_mask):
    """Calculate Dice coefficient"""
    pred_binary = (pred_mask > 127).astype(np.uint8)
    gt_binary = (gt_mask > 127).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    pred_sum = pred_binary.sum()
    gt_sum = gt_binary.sum()
    
    if pred_sum + gt_sum == 0:
        return 0.0
    
    return 2 * intersection / (pred_sum + gt_sum)


def main():
    print("Naive Baseline: Canny Edge Detection")
    print("="*80)
    
    # Get test images (INPUT_DIR 사용)
    image_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')])
    
    results = []
    total_iou = 0
    total_dice = 0
    
    for img_file in tqdm(image_files, desc='Processing'):
        img_path = os.path.join(INPUT_DIR, img_file)
        
        # Baseline prediction
        pred_mask, original = canny_baseline(img_path)
        
        # Load ground truth
        mask_file = img_file.replace('.jpg', '.png')
        gt_path = os.path.join(GT_MASK_DIR, mask_file)
        
        if not os.path.exists(gt_path):
            print(f"Warning: GT mask not found for {img_file}")
            continue
        
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Calculate metrics
        iou = calculate_iou(pred_mask, gt_mask)
        dice = calculate_dice(pred_mask, gt_mask)
        total_iou += iou
        total_dice += dice
        
        results.append({
            'image': img_file,
            'iou': float(iou),
            'dice': float(dice),
            'pred_pixels': int(np.sum(pred_mask > 127)),
            'gt_pixels': int(np.sum(gt_mask > 127))
        })
        
        # Save mask
        mask_path = os.path.join(OUTPUT_DIR, 'masks', mask_file)
        cv2.imwrite(mask_path, pred_mask)
        
        # Save overlay
        overlay = original.copy()
        overlay[pred_mask > 127] = [0, 255, 0]  # Green for baseline
        overlay[gt_mask > 127] = [255, 0, 0]    # Red for GT
        
        overlay_path = os.path.join(OUTPUT_DIR, 'overlays', img_file)
        cv2.imwrite(overlay_path, overlay)
    
    # Calculate averages
    avg_iou = total_iou / len(results) if results else 0
    avg_dice = total_dice / len(results) if results else 0
    
    # Save detailed results
    summary = {
        'method': 'Canny Edge Detection (Naive Baseline)',
        'num_images': len(results),
        'average_iou': float(avg_iou),
        'average_dice': float(avg_dice),
        'per_image_results': results
    }
    
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print comparison table format
    print("\n" + "="*80)
    print("Baseline Results")
    print("="*80)
    print(f"Number of images: {len(results)}")
    print(f"Average IoU:  {avg_iou:.4f}")
    print(f"Average Dice: {avg_dice:.4f}")
    print("\n" + "="*80)
    print("Comparison Table (for report)")
    print("="*80)
    print(f"| Method                | IoU    | Dice   |")
    print(f"|----------------------|--------|--------|")
    print(f"| Canny Baseline       | {avg_iou:.4f} | {avg_dice:.4f} |")
    print(f"| U-Net                | 0.2195 | 0.3145 |")  #  Our U-Net results
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"- Numerical results: {os.path.join(OUTPUT_DIR, 'results.json')}")
    print(f"- Overlay images: {os.path.join(OUTPUT_DIR, 'overlays/')}")
    print(f"- Predicted masks: {os.path.join(OUTPUT_DIR, 'masks/')}")


if __name__ == "__main__":
    main()
