"""Hugging Face text generation helpers."""

from __future__ import annotations

from functools import lru_cache

from utils.config import settings


PROMPT_TEMPLATE = (
    "You are a precise customer support assistant. "
    "Answer directly, use only the provided context when it is available, and keep the response concise.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)


@lru_cache(maxsize=1)
def get_generator(model_name: str | None = None):
    """Load and cache the sequence-to-sequence tokenizer and model."""

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("transformers is required for response generation") from exc

    resolved_name = model_name or settings.llm_model_name
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(resolved_name)
    return tokenizer, model


def generate_response(question: str, context: str = "", model_name: str | None = None) -> str:
    """Generate an answer for a question with optional retrieved context."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("torch is required for response generation") from exc

    normalized_question = question.strip()
    if not normalized_question:
        raise ValueError("question must not be empty")

    tokenizer, model = get_generator(model_name)
    normalized_context = context.strip() or "No supporting context provided."
    prompt = PROMPT_TEMPLATE.format(context=normalized_context, question=normalized_question)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    with torch.no_grad():
        output_tokens = model.generate(
            **encoded,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False,
        )
    decoded = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return decoded.strip()
