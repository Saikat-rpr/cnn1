import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


from models.model_v1 import SimpleCNN   
from models.model_v2 import SimpleCNNv2     


def find_last_conv_layer(model: nn.Module):
    """
    Automatically find the last nn.Conv2d layer in the model.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in the model.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

        
        self.target_layer = find_last_conv_layer(self.model)

        self.activations = None
        self.gradients = None

        
        self.fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int):
        """
        input_tensor: [1, C, H, W]
        target_class: int (index of class to explain)
        Returns: heatmap as numpy array [H, W] normalized 0..1
        """
        self.model.zero_grad()

        
        output = self.model(input_tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]

        
        score = output[:, target_class]

        
        score.backward(retain_graph=True)

        # Grad-CAM computation
        # gradients: [B, C, H', W']
        # activations: [B, C, H', W']
        gradients = self.gradients        # [1, C, H', W']
        activations = self.activations    # [1, C, H', W']

        
        weights = gradients.mean(dim=(2, 3))[0]  

        
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32).to(activations.device)
        for c, w in enumerate(weights):
            cam += w * activations[0, c, :, :]

        
        cam = torch.relu(cam)

        # Normalize to 0..1
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )[0, 0]

        return cam.detach().cpu().numpy()

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def tensor_to_pil(t):
    # t: [C,H,W] 0..1
    t = t.detach().cpu().clamp(0, 1)
    np_img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_img)


def overlay_heatmap_on_image(img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5):
    """
    img: PIL image (RGB)
    heatmap: [H,W] numpy (0..1)
    alpha: blending factor
    """
    
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]  # drop alpha
    colored = (colored * 255).astype(np.uint8)
    heatmap_img = Image.fromarray(colored).resize(img.size, resample=Image.BILINEAR)

    
    return Image.blend(img.convert("RGB"), heatmap_img, alpha=alpha)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualizations for model_v1 and model_v2")
    parser.add_argument("--model", type=str, choices=["v1", "v2"], required=True,
                        help="Which model to use: v1 (Animals) or v2 (PlantVillage)")
    parser.add_argument("--weights_v1", type=str, default="models/model_v1.pth",
                        help="Path to model_v1 weights")
    parser.add_argument("--weights_v2", type=str, default="models/model_v2.pth",
                        help="Path to model_v2 weights")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to an ImageFolder dataset (e.g., val split)")
    parser.add_argument("--output_dir", type=str, default="results/gradcam",
                        help="Where to save Grad-CAM images")
    parser.add_argument("--num_images", type=int, default=8,
                        help="Number of images to visualize")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")
    args = parser.parse_args()

    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    if args.model == "v1":
        print("Loading model_v1 (Animals)...")
        model = SimpleCNN(num_classes=5)  
        state = torch.load(args.weights_v1, map_location=device)
        model.load_state_dict(state)
    else:
        print("Loading model_v2 (PlantVillage)...")
        model = SimpleCNNv2(num_classes=5)   
        state = torch.load(args.weights_v2, map_location=device)
        model.load_state_dict(state)

    model.to(device)
    model.eval()

    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])

    
    dataset = ImageFolder(args.data_dir, transform=transform)
    class_names = dataset.classes
    print(f"Found {len(dataset)} images across {len(class_names)} classes in {args.data_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    gradcam = GradCAM(model)

    
    num_images = min(args.num_images, len(dataset))
    for idx in range(num_images):
        img_tensor, label = dataset[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)  # [1,C,H,W]

        
        with torch.no_grad():
            logits = model(img_tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()

        
        target_class = pred_class
        cam = gradcam.generate(img_tensor, target_class)

        
        pil_img = tensor_to_pil(img_tensor[0])

        
        overlay = overlay_heatmap_on_image(pil_img, cam, alpha=0.5)

        # Save side-by-side visualization
        # left: original, right: overlay
        combined = Image.new("RGB", (pil_img.width * 2, pil_img.height))
        combined.paste(pil_img, (0, 0))
        combined.paste(overlay, (pil_img.width, 0))

        out_name = f"{args.model}_gradcam_{idx:03d}_pred-{class_names[pred_class]}_label-{class_names[label]}.png"
        out_path = os.path.join(args.output_dir, out_name)
        combined.save(out_path)
        print(f"Saved: {out_path}")

    gradcam.close()
    print("Done.")


if __name__ == "__main__":
    main()
