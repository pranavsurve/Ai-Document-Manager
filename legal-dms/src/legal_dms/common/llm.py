"""LLM client abstractions and provider implementations for Legal DMS."""

import json
from typing import Any, Iterator, Protocol

import anthropic
import openai
import requests
from pydantic import BaseModel, ValidationError

from legal_dms.config.settings import settings


class LlmSchemaError(Exception):
    def __init__(
        self,
        first_output: str,
        first_error: str,
        second_output: str,
        second_error: str,
    ):
        self.attempts = [
            {"output": first_output, "error": first_error},
            {"output": second_output, "error": second_error},
        ]
        super().__init__(
            "LLM response failed schema validation twice."
        )


class LlmClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
    ) -> Any:
        ...

    def stream(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        ...


def _schema_instructions(response_schema: type[BaseModel]) -> str:
    schema_json = response_schema.model_json_schema()
    return (
        "Return only valid JSON that matches the following pydantic schema exactly, "
        "without any additional text or markdown formatting:\n"
        f"{json.dumps(schema_json, indent=2, ensure_ascii=False)}"
    )


def _validate_schema_output(
    output: str, response_schema: type[BaseModel]
) -> BaseModel:
    try:
        return response_schema.model_validate_json(output)
    except ValidationError as error:
        raise


class BaseLlmClient:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key

    def complete(
        self,
        system: str,
        user: str,
        response_schema: type[BaseModel] | None = None,
        temperature: float = 0.2,
    ) -> Any:
        if response_schema is None:
            return self._complete(system, user, temperature)

        instruction = _schema_instructions(response_schema)
        first_output = self._complete(system, f"{user}\n\n{instruction}", temperature)
        try:
            return _validate_schema_output(first_output, response_schema)
        except ValidationError as first_error:
            retry_user = (
                f"{user}\n\n"
                "The previous response failed schema validation."
                f"\nValidation error: {first_error}\n"
                f"Original output: {first_output}\n"
                "Please reply with only valid JSON matching the schema."
            )
            second_output = self._complete(system, retry_user, temperature)
            try:
                return _validate_schema_output(second_output, response_schema)
            except ValidationError as second_error:
                raise LlmSchemaError(
                    first_output=str(first_output),
                    first_error=str(first_error),
                    second_output=str(second_output),
                    second_error=str(second_error),
                )

    def stream(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        return self._stream(system, user, temperature)

    def _complete(self, system: str, user: str, temperature: float) -> str:
        raise NotImplementedError

    def _stream(self, system: str, user: str, temperature: float) -> Iterator[str]:
        raise NotImplementedError


class AnthropicClient(BaseLlmClient):
    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        self.client = anthropic.Client(api_key=api_key)

    def _complete(self, system: str, user: str, temperature: float) -> str:
        prompt = f"{anthropic.HUMAN_PROMPT}{system}\n\n{user}{anthropic.AI_PROMPT}"
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
        )
        return str(response.completion)

    def _stream(self, system: str, user: str, temperature: float) -> Iterator[str]:
        prompt = f"{anthropic.HUMAN_PROMPT}{system}\n\n{user}{anthropic.AI_PROMPT}"
        stream = self.client.completions.stream(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
        )
        for event in stream:
            if hasattr(event, "completion"):
                yield str(event.completion)
            elif isinstance(event, dict) and event.get("type") == "response.output_text.delta":
                yield str(event.get("delta", ""))


class OpenAiClient(BaseLlmClient):
    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model, api_key)
        openai.api_key = api_key

    def _complete(self, system: str, user: str, temperature: float) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return str(response.choices[0].message.content)

    def _stream(self, system: str, user: str, temperature: float) -> Iterator[str]:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta.get("content")
            if delta:
                yield str(delta)


class OllamaClient(BaseLlmClient):
    def __init__(self, model: str, api_key: str | None = None) -> None:
        super().__init__(model, api_key)
        self.base_url = settings.ollama_url.rstrip("/")

    def _complete(self, system: str, user: str, temperature: float) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions", json=payload, timeout=30
        )
        response.raise_for_status()
        body = response.json()
        return str(body["choices"][0]["message"]["content"])

    def _stream(self, system: str, user: str, temperature: float) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": True,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions", json=payload, stream=True, timeout=30
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line.removeprefix("data:").strip())
            token = payload.get("text") or payload.get("choices", [{}])[0].get("delta", {}).get("content")
            if token:
                yield str(token)


_client_cache: dict[str, LlmClient] = {}


def get_llm_client() -> LlmClient:
    key = f"{settings.llm_provider.lower()}::{settings.llm_model}"
    if key in _client_cache:
        return _client_cache[key]

    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        client = AnthropicClient(settings.llm_model, settings.llm_api_key)
    elif provider == "openai":
        client = OpenAiClient(settings.llm_model, settings.llm_api_key)
    elif provider == "ollama":
        client = OllamaClient(settings.llm_model)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    _client_cache[key] = client
    return client
