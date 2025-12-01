import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from .base_model import BaseVQAModel


class BLIP2VQAModel(BaseVQAModel):
    """BLIP-2 for VQA with generative answering"""

    def __init__(self, model_size="blip2-opt-2.7b"):
        super().__init__(f"BLIP2_{model_size}")
        self.model_size = model_size
        self.model = None
        self.processor = None

    def load_model(self):
        print(f"Loading {self.model_name}")
        model_path = f"Salesforce/{self.model_size}"
        self.processor = Blip2Processor.from_pretrained(model_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.eval()
        print(f"{self.model_name} ready to use")

    def predict(self, image_path: str, question: str) -> str:
        """Generate answer using BLIP-2"""
        assert self.model is not None, "Model not loaded"
        assert self.processor is not None, "Processor not loaded"

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image,
            text=f"Question: {question}",
            return_tensors="pt",  # type:ignore
        ).to(self.model.device, torch.float16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=20, min_length=1, num_beams=5, temperature=1.0
            )

        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        answer = answer.strip()

        return answer

    def cleanup(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
