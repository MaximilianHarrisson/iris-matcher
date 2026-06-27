import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class InceptionV3Extractor:
    """https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.inception_v3.html"""
    INPUT_SIZE = 299

    def __init__(self):
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(resolved_device)

        self.model = self._initialize_inceptionv3()
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _initialize_inceptionv3(self) -> nn.Module:
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        model.fc = nn.Identity()
        model = model.to(self.device)
        model.eval()
        return model

    def _load_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f'File not found: {img_path}')

        img = cv2.resize(img, (self.INPUT_SIZE, self.INPUT_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def _tensorize_image(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return (img_tensor - self.mean) / self.std

    def _extract_embedding(self, img_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()

        with torch.no_grad():
            features = self.model(img_tensor)
            embedding = features.flatten(1).squeeze(0).detach().cpu().numpy()

        return embedding

    def extract(self, img_path: str) -> np.ndarray:
        try:
            img = self._load_image(img_path)
            img_tensor = self._tensorize_image(img)
            embedding = self._extract_embedding(img_tensor)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f'Failed to extract InceptionV3 features from: {img_path}') from exc

        return embedding
