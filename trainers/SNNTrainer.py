import sys, os

import torch
import transformers

from transformers import Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.configuration_SNN import SNN

class SNNTrainer(Trainer):
    def compute_loss(self, model, batch) -> torch.Tensor:
        def ensure_batch_dimension(tensor):
            """Ensure the tensor has a batch dimension."""
            if tensor is not None and tensor.dim() == 3:
                return tensor.unsqueeze(0) 
            return tensor

        input_images = ensure_batch_dimension(batch['input_images'])
        target_images = ensure_batch_dimension(batch['target_images'])
        negative_images = ensure_batch_dimension(batch.get('negative_images'))

        # Compute loss using the model
        return model(input_images, target_images, negative_images)