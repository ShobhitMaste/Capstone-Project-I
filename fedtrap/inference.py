import os
import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Union, Any
from fedtrap.model import load_model, create_mobilenetv2
from fedtrap.dataset import get_transforms
from fedtrap.config import get_class_names, get_class_category, get_class_action, get_config

class InsectClassifier:
    def __init__(self, model_path: str, config_path: str = None, device: str = None):
        self.config = get_config() if config_path is None else get_config(config_path)
        self.class_names = get_class_names()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.is_onnx = model_path.endswith('.onnx')
        if self.is_onnx:
            import onnxruntime as ort
            self.ort_session = ort.InferenceSession(model_path)
        else:
            self.model = create_mobilenetv2(num_classes=len(self.class_names), pretrained=False)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
        self.transform = get_transforms(train=False)

    def preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        return self.transform(image).unsqueeze(0)

    def predict(self, image: Union[str, Image.Image]) -> Dict[str, Any]:
        tensor = self.preprocess(image)
        if self.is_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: tensor.numpy()}
            ort_outs = self.ort_session.run(None, ort_inputs)
            output = torch.tensor(ort_outs[0])
        else:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                output = self.model(tensor)
                
        probs = torch.nn.functional.softmax(output[0], dim=0).cpu().numpy()
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        class_name = self.class_names[pred_idx]
        
        return {
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': {name: float(prob) for name, prob in zip(self.class_names, probs)},
            'category': get_class_category(class_name),
            'action': get_class_action(class_name)
        }

    def predict_batch(self, images: List[Union[str, Image.Image]]) -> List[Dict[str, Any]]:
        return [self.predict(img) for img in images]
