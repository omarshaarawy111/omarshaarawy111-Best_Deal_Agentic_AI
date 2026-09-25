# Deoploy modal app for fine-tuned model 
# Fine tuning prediction: fine-tuned model ID ---> tokenization ---> quantization ---> load fine-tuned model ---> prediction
import modal
from modal import Volume, Image

app = modal.App("best-deal-fine-tuned-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

secrets = [modal.Secret.from_name("huggingface-secret")]

# Configurations
GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
# The constant project name
PROJECT_NAME = "best-deal-price"
USERNAME = "omarshaarawy11"
HUB_MODEL_NAME = f"{USERNAME}/best-deal-2026-09-10_18.41.06"
SNAPSHOT = "776421863beaeccbfd320c41b69c7fab10095ca6"
CACHE_DIR = "/cache"

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
# MIN_CONTAINERS = 0 for safely low cost
MIN_CONTAINERS = 0

PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

# Script to be run on Modal 
@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        # Tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        # Quantization
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto"
        )
        # Load fine-tuned model 
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, HUB_MODEL_NAME, revision=SNAPSHOT
        )

    # Prediction 
    @modal.method()
    def price(self, description: str) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output_ids = self.fine_tuned_model.generate(
                **inputs,
                max_new_tokens=3
            )
        prompt_len = inputs["input_ids"].shape[1]
        # Trim output
        generated_ids = output_ids[0, prompt_len:]
        # Decode output
        output = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        # Extract output's number only
        match = re.search(r"\d+(?:\.\d+)?", output)
        return float(match.group()) if match else 0.0