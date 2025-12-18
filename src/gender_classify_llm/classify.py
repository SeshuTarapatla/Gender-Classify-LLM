from json import loads
from pathlib import Path
from typing import Literal, TypedDict

from rich.status import Status

from .ollama import ollama


SYSTEM_PROMPT = """
You are a strict binary classification system.

Task:
Estimate the most likely gender using the provided profile image and text.
This is a probabilistic inference, not ground truth.

Output rules (MANDATORY):
- Output MUST be valid JSON
- Output MUST contain ONLY these keys:
  - gender
  - confidence
  - reason
- gender MUST be an string with the below possible values only:
  - female
  - male
- confidence MUST be a float between 0 and 1
- reason MUST be a string with a single line describing:
  - why do you think it is that gender
- No explanations
- No extra keys
- No text outside JSON
- No markdown
- No comments

Behavior rules:
- Treat this as a classification task, not a reasoning task
- Do NOT perform or reveal step-by-step reasoning
- Minimize internal deliberation
- Respond decisively within constraints
- Never ask questions
- Never apologize
- If output violates rules, immediately correct and output valid JSON only
- Be as quick as possible to deliver the output
"""


class Prediction(TypedDict):
    gender: Literal["male", "female"]
    confidence: float
    reason: str


class GenderClassifier:
    def __init__(self, model: str = "qwen3-vl:4b-instruct") -> None:
        
        self.model = model
        self.__system_prompt__ = SYSTEM_PROMPT
        ollama.exists(self.model)
        with Status(f"Loading the model [green]{self.model}[/]..."):
            self._generate("Hi")

    def _generate(self, prompt: str, image: str | Path | None = None) -> str:
        return ollama.generate(
            model=self.model,
            system=self.__system_prompt__,
            images=[str(image)] if image else None,
            prompt=prompt,
        ).response

    def predict(self, image: str | Path) -> Prediction:
        response = self._generate(
            prompt="Predict the gender from the image provided.", image=image
        )
        result: Prediction = loads(response)
        return result
