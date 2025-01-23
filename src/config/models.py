from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["LM_STUDIO_API_BASE"] = "http://127.0.0.1:1234/v1"


@dataclass
class ModelConfig:
    name: str
    api_key: str


models = {
    "claude-3.5": ModelConfig(
        name="anthropic/claude-3-5-sonnet-latest",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    ),
    "mistral": ModelConfig(
        name="openrouter/mistralai/pixtral-12b",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "llama": ModelConfig(
        name="openrouter/meta-llama/llama-3.2-90b-vision-instruct",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "llama 11B": ModelConfig(
        name="openrouter/meta-llama/llama-3.2-11b-vision-instruct",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "llava": ModelConfig(
        name="openrouter/fireworks/firellava-13b",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "Grok Vision Beta": ModelConfig(
        name="openrouter/x-ai/grok-vision-beta",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "Qwen2-VL 7B Instruct": ModelConfig(
        name="openrouter/qwen/qwen-2-vl-7b-instruct",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "Gemini 1.5 Flash-8B": ModelConfig(
        name="openrouter/google/gemini-flash-1.5-8b",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "GPT-4o": ModelConfig(
        name="openrouter/openai/gpt-4o-2024-11-20",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "Gemini Pro 1.5 Experimental": ModelConfig(
        name="openrouter/google/gemini-pro-1.5-exp",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "Gemini Pro 1.5": ModelConfig(
        name="gemini/gemini-1.5-pro",
        api_key=os.environ["GOOGLE_API_KEY"],
    ),
    "Phi-3.5-vision": ModelConfig(name="lm_studio/phi-3.5-vision-instruct", api_key=""),
    "Qwen2-VL 72B Instruct": ModelConfig(
        name="openrouter/qwen/qwen-2-vl-72b-instruct",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
    "Qwen2-VL 7B Instruct": ModelConfig(
        name="openrouter/qwen/qwen-2-vl-7b-instruct",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ),
}
