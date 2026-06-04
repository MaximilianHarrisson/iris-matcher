import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class Resnet50Extractor:
    """https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.resnet50.html"""
    def __init__(self):
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(resolved_device)

        self.feature_extractor, self.pooling = self._initialize_resnet50()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _initialize_resnet50(self) -> tuple:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        feature_extractor = nn.Sequential(*list(model.children())[:-2]).to(self.device)
        pooling = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        feature_extractor.eval()
        pooling.eval()
        return feature_extractor, pooling

    def _load_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'File not found: {img_path}')

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def _tensorize_image(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return (img_tensor - self.mean) / self.std

    def _extract_embedding(self, img_tensor: torch.Tensor) -> np.ndarray:
        self.feature_extractor.eval()
        self.pooling.eval()

        with torch.no_grad():
            features = self.feature_extractor(img_tensor)
            pooled = self.pooling(features).flatten(1)
            embedding = pooled.squeeze(0).detach().cpu().numpy()

        return embedding

    def extract(self, img_path: str) -> np.ndarray:
        try:
            img = self._load_image(img_path)
            img_tensor = self._tensorize_image(img)
            embedding = self._extract_embedding(img_tensor)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f'Failed to extract ResNet50 features from: {img_path}') from exc

        return embedding
