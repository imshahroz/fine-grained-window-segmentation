import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from model import SimpleSegNet
from utils import refine_mask


def run_inference(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    model_path = "outputs/window_seg_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleSegNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(input_tensor)[0, 0].cpu().numpy()

    refined = refine_mask(prediction)

    original_resized = cv2.resize(original, (256, 256))
    cleaned_output = cv2.bitwise_and(original_resized, original_resized, mask=refined)

    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite("outputs/predicted_mask.png", refined)
    cv2.imwrite("outputs/cleaned_output.png", cv2.cvtColor(cleaned_output, cv2.COLOR_RGB2BGR))

    print("Saved outputs:")
    print("outputs/predicted_mask.png")
    print("outputs/cleaned_output.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    run_inference(args.image)
