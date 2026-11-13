"""Thin wrapper around a seq2seq instruction-tuned model (FLAN-T5)."""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Generator:
    def __init__(self, name: str = "google/flan-t5-large", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.device).eval()

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.9,
        top_p: float = 0.95,
        max_new_tokens: int = 80,
        seed: int | None = None,
    ) -> list[str]:
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
        )
        return [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
