import os

import torch

from .base_model import BaseVQAModel
from .vlm_chatbot.models.vlm import VLMChatbot


class myModelVQAModel(BaseVQAModel):
    """Custom Trained VLM ChatBot for VQA"""

    def __init__(
        self,
        checkpoint_path="models/vlm_chatbot/checkpoints/vlm_projection_best_improved.pth",
        load_in_4bit=True,
        temperature=0.2,
    ):
        super().__init__("VLM_ChatBot")
        self.checkpoint_path = checkpoint_path
        self.load_in_4bit = load_in_4bit
        self.temperature = temperature
        self.model = None

    def load_model(self):
        """Load the VLM ChatBot model"""
        print(f"Loading {self.model_name}")

        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            print(f"ERROR: Checkpoint not found at {self.checkpoint_path}")
            checkpoint_to_load = None
        else:
            checkpoint_to_load = self.checkpoint_path

        self.model = VLMChatbot(
            load_in_4bit=self.load_in_4bit,
            checkpoint_path=checkpoint_to_load,
        )
        self.model.eval()
        print(f"{self.model_name} ready to use")

    def predict(self, image_path: str, question: str) -> str:
        """Generate answer using VLM ChatBot"""
        assert self.model is not None, "Model not loaded"

        with torch.no_grad():
            response = self.model.chat(
                image_path=image_path,
                question=question,
                max_new_tokens=50,
                temperature=self.temperature,
            )

        if not response or len(response.strip()) == 0:
            return "[Model generated empty response]"

        return response.strip()

    def cleanup(self):
        """Clean up model resources"""
        if self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
