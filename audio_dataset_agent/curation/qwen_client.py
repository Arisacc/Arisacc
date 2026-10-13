class TransformersQwenClient:
    """基于 Transformers 运行时的本地 Qwen 文本客户端。"""

    def __init__(
        self,
        model_path: str,
        tokenizer=None,
        model=None,
        max_new_tokens: int = 128,
        device_map: str = "auto",
        torch_dtype: str = "auto",
    ):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self.torch_dtype = torch_dtype

    def complete(self, prompt: str) -> str:
        tokenizer, model = self._ensure_loaded()
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([rendered], return_tensors="pt")
        target_device = getattr(model, "device", None)
        if target_device is not None:
            inputs = {
                name: value.to(target_device) if hasattr(value, "to") else value
                for name, value in inputs.items()
            }
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        input_length = inputs["input_ids"].shape[1]
        return tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True,
        ).strip()

    def _ensure_loaded(self):
        if self.tokenizer is not None and self.model is not None:
            return self.tokenizer, self.model

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            local_files_only=True,
        )
        return self.tokenizer, self.model
