"""Thin OpenAI-compatible vision client for a vLLM server (build-plan step 3).

ONE adapter drives every model — only base_url + served id change — so the harness is
byte-identical across models. Lifted from src_legacy_project_for_ref/generation/vlm_client.py
(b64 encode :56-61, message build :63-84, retry/backoff :87-119); adds seed/sampling,
multi-image messages, a downscale-before-encode step, and a preflight check.
"""
from __future__ import annotations

import time
import base64
import logging
from io import BytesIO
from dataclasses import dataclass

from PIL import Image

log = logging.getLogger(__name__)
CALL_FAILED = "[VLM_CALL_FAILED]"


@dataclass
class ChatResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    ok: bool


def encode_image(img: Image.Image, fmt: str = "JPEG", quality: int = 95, max_side: int | None = 1024) -> str:
    """PIL image -> data-URL string for an OpenAI image_url content block."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max_side:
        w, h = img.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = BytesIO()
    save_kw = {"quality": quality} if fmt.upper() == "JPEG" else {}
    img.save(buf, format=fmt, **save_kw)
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def image_block(data_url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url}}


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


class OpenAIVisionClient:
    def __init__(self, base_url, api_key="EMPTY", model="", timeout_s=120, retries=3):
        self.base_url, self.api_key, self.model = base_url, api_key, model
        self.timeout_s, self.retries = timeout_s, retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai
            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key,
                                         timeout=self.timeout_s)
        return self._client

    def list_models(self):
        return [m.id for m in self.client.models.list().data]

    def chat(self, messages, seed, temperature, top_p, max_tokens, extra_body=None) -> ChatResponse:
        body = {"seed": seed}
        if extra_body:
            body.update(extra_body)
        last = None
        for attempt in range(self.retries):
            try:
                t0 = time.perf_counter()
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, seed=seed,
                    extra_body=body,
                )
                u = resp.usage
                return ChatResponse(
                    content=(resp.choices[0].message.content or "").strip(),
                    prompt_tokens=u.prompt_tokens if u else 0,
                    completion_tokens=u.completion_tokens if u else 0,
                    latency_s=time.perf_counter() - t0, ok=True)
            except Exception as e:  # network / server / rate-limit
                last = e
                log.warning("chat attempt %d failed: %s", attempt + 1, e)
                time.sleep(2 ** attempt)
        log.error("chat failed after %d attempts: %s", self.retries, last)
        return ChatResponse(CALL_FAILED, 0, 0, 0.0, ok=False)
