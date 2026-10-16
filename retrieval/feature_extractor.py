import torch
import numpy as np
from PIL import Image

# Model is loaded lazily on first use (ResNet50 costs ~200MB of RAM)
model = None
transform = None


def load_model():
    global model, transform
    if model is None:
        import torchvision.transforms as transforms
        from torchvision.models import resnet50, ResNet50_Weights

        m = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(m.children())[:-1])
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225]
            )
        ])
    return model, transform


def extract_features(image_path: str) -> np.ndarray:
    m, tf = load_model()
    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0)

    with torch.no_grad():
        features = m(tensor)

    vector = features.squeeze().numpy()
    vector = vector / np.linalg.norm(vector)

    return vector
