from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


def extract_nonstream_text(response) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    for attr in ["content", "output_text", "reasoning_content", "reasoning"]:
        value = getattr(message, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = []
            for block in value:
                for key in ["text", "content", "output_text"]:
                    v = block.get(key) if isinstance(block, dict) else getattr(block, key, None)
                    if isinstance(v, str) and v.strip():
                        parts.append(v)
            if parts:
                return "".join(parts).strip()
    return ""


def extract_stream_text(stream_response) -> str:
    parts = []
    for chunk in stream_response:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            parts.append(content)
    return "".join(parts).strip()


@dataclass
class OpenAIChatModel:
    name: str
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    stream: bool = False
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def list_models(self):
        return self.client.models.list()

    def call(self, system_prompt: str, user_prompt: str):
        if self.stream:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
                extra_body=self.extra_body or None,
            )
            return "stream", extract_stream_text(response)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            extra_body=self.extra_body or None,
        )
        finish_reason = getattr(response.choices[0], "finish_reason", None)
        return finish_reason, extract_nonstream_text(response)


@dataclass
class LocalPathChatModel:
    name: str
    path: str
    model: str
    temperature: float
    max_tokens: int
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Local model inference requires transformers and torch. "
                "Install dependencies from requirements.txt."
            ) from e

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.path,
            trust_remote_code=True,
            use_fast=False,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype="auto",
        )
        self._model.eval()

    def list_models(self):
        class _ModelItem:
            def __init__(self, model_id: str):
                self.id = model_id

        class _ModelList:
            def __init__(self, model_id: str):
                self.data = [_ModelItem(model_id)]

        return _ModelList(self.model)

    def _build_inputs(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self._tokenizer(prompt, return_tensors="pt")

        prompt = f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
        return self._tokenizer(prompt, return_tensors="pt")

    def call(self, system_prompt: str, user_prompt: str):
        inputs = self._build_inputs(system_prompt, user_prompt)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        do_sample = self.temperature > 0
        generation_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": do_sample,
            "temperature": self.temperature if do_sample else None,
            "top_p": 0.95 if do_sample else None,
            "pad_token_id": getattr(self._tokenizer, "pad_token_id", None) or getattr(self._tokenizer, "eos_token_id", None),
            "eos_token_id": getattr(self._tokenizer, "eos_token_id", None),
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        generation_config = getattr(self._model, "generation_config", None)
        if generation_config is not None and not do_sample:
            # Some local models ship sampling defaults in generation_config.
            # Override them explicitly to avoid warnings and unstable behavior.
            generation_config.do_sample = False
            if hasattr(generation_config, "temperature"):
                generation_config.temperature = None
            if hasattr(generation_config, "top_p"):
                generation_config.top_p = None

        with self._torch.no_grad():
            outputs = self._model.generate(**inputs, **generation_kwargs)

        generated = outputs[0][input_length:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return "local_generate", text.strip()


def build_model(model_key: str, model_registry: Dict[str, Dict[str, Any]], temperature: float, max_tokens: int):
    cfg = model_registry[model_key]
    if cfg["source_type"] == "local_path":
        path = cfg["path"]
        if not Path(path).exists():
            raise FileNotFoundError(f"Local model path does not exist: {path}")
        return LocalPathChatModel(
            name=model_key,
            path=path,
            model=cfg["model"],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=cfg.get("extra_body", {}),
        )

    return OpenAIChatModel(
        name=model_key,
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=bool(cfg.get("stream", False)),
        extra_body=cfg.get("extra_body", {}),
    )
