import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base_model import BaseVQAModel


class CLIPVQAModel(BaseVQAModel):
    """
    CLIP for VQA using zero-shot classification

    Note: CLIP doesn't generate text, so convert VQA to classification by providing candidate answers
    """

    def __init__(self):
        super().__init__("CLIP")
        self.model = None
        self.processor = None

    def load_model(self):
        """Load Model"""
        print(f"Loading {self.model_name}")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.model.to(self.device)  # type: ignore
        self.model.eval()
        print(f"{self.model_name} ready to use")

    def predict(self, image_path: str, question: str) -> str:
        """Use CLIP for VQA by"""
        assert self.model is not None, "Model not loaded"
        assert self.processor is not None, "Processor not loaded"

        image = Image.open(image_path).convert("RGB")
        candidates = self._get_candidate_answers(question)
        texts = [f"Q: {question} A: {answer}" for answer in candidates]

        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",  # type: ignore
            padding=True,  # type: ignore
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        best_idx = probs.argmax().item()
        return candidates[best_idx]

    def _get_candidate_answers(self, question: str) -> list:
        """Generate candidate answers based on question"""
        question_lower = question.lower()

        # Boolean
        if any(
            word in question_lower
            for word in ["is", "are", "does", "do", "can", "will"]
        ):
            return ["yes", "no"]

        # Counting
        if "how many" in question_lower:
            return ["0", "1", "2", "3", "4", "5", "many", "several"]

        # Color
        if "color" in question_lower or "what color" in question_lower:
            return [
                "red",
                "blue",
                "green",
                "yellow",
                "black",
                "white",
                "brown",
                "gray",
                "orange",
                "purple",
            ]

        # Default: common VQA answers
        return [
            "yes",
            "no",
            "person",
            "people",
            "man",
            "woman",
            "child",
            "dog",
            "cat",
            "bird",
            "car",
            "bus",
            "truck",
            "train",
            "table",
            "chair",
            "bed",
            "couch",
            "tree",
            "building",
            "red",
            "blue",
            "green",
            "white",
            "black",
            "1",
            "2",
            "3",
            "many",
        ]

    def cleanup(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
