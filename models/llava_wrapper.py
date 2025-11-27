import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor

from .base_model import BaseVQAModel


class LLaVAVQAModel(BaseVQAModel):
    """LLaVA for VQA with instruction following"""

    def __init__(self, model_size="7b"):
        super().__init__(f"LLaVA_{model_size}")
        self.model_size = model_size
        self.model = None
        self.tokenizer = None
        self.image_processor = None

    def load_model(self):
        print(f"Loading {self.model_name}")

        model_path = f"liuhaotian/llava-v1.5-{self.model_size}"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        self.image_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        self.model.eval()
        print(f"{self.model_name} ready to use")

    def predict(self, image_path: str, question: str) -> str:
        """Generate answer using LLaVA"""
        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Processor not loaded"
        assert self.image_processor is not None, "Image Processor not loaded"

        image = Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        image_tensor = (
            self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            .half()
            .to(self.model.device)
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            self.model.device
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=50,
                temperature=0.2,
                do_sample=True,
                use_cache=True,
            )

        answer = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        return answer

    def cleanup(self):
        del self.model
        del self.tokenizer
        del self.image_processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
