from openai import OpenAI, OpenAIError

from . import state
from .config import MAX_OUTPUT_TOKENS, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from .memory import memory_context
from .prompts import ACTION_CLASSIFIER_PROMPT, ASSISTANT_PROMPT
from .text_utils import clean_assistant_output


def lm_client():
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def usage_dict(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        name: getattr(usage, name)
        for name in ("input_tokens", "output_tokens", "total_tokens")
        if hasattr(usage, name)
    }


def debug_response(kind, prompt, response):
    state.debug(f"{kind} model", OPENAI_MODEL)
    state.debug(f"{kind} prompt", prompt)
    state.debug(f"{kind} raw output", getattr(response, "output_text", ""))
    state.debug(f"{kind} usage", usage_dict(response) or "not reported by server")


def ask_model(text):
    prompt = (
        f"{ASSISTANT_PROMPT}\n"
        f"Saved memory:\n{memory_context()}\n\n"
        f"Answer in English.\n"
        f"User: {text}"
    )
    try:
        response = lm_client().responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    except OpenAIError as exc:
        state.debug("chat error", str(exc))
        return "I cannot reach LM Studio right now. Start the local server and try again."
    debug_response("chat", prompt, response)
    return clean_assistant_output(response.output_text)


def classify_action(text):
    prompt = f"{ACTION_CLASSIFIER_PROMPT}\nUser: {text}"
    try:
        response = lm_client().responses.create(model=OPENAI_MODEL, input=prompt, max_output_tokens=40)
    except OpenAIError as exc:
        state.debug("action error", str(exc))
        return "chat"
    debug_response("action", prompt, response)
    return response.output_text.strip().splitlines()[0].strip()
