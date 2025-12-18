from json import loads
from pathlib import Path
from typing import Literal, TypedDict

from rich.status import Status

from .ollama import ollama


# Defining the system prompt for the LLM for gender classification.
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
    """
    Represents the prediction result from a gender classifier.

    Attributes:
        gender (Literal["male", "female"]): The predicted gender of the subject.
        confidence (float): The model's confidence score (0.0 to 1.0).
        reason (str): A textual explanation or reasoning for the prediction.
    """

    gender: Literal["male", "female"]
    confidence: float
    reason: str


class GenderClassifier:
    """
    A class for classifying gender using a ollama vision language model.
    This class provides methods to process image & text and predict gender based on visual and  linguistic patterns.
    """

    def __init__(self, model: str = "qwen3-vl:4b-instruct") -> None:
        """
        Initializes the GenderClassifier with an Ollama model.

        Args:
            model (str): The Ollama model name to use. Defaults to "**qwen3-vl:4b-instruct**".

        Raises:
            Exception: If the model or ollama does not exist.
        """

        self.model = model
        self.__system_prompt__ = SYSTEM_PROMPT
        ollama.exists(self.model)
        if not ollama.is_running(self.model):
            with Status(f"Loading the model [green]{self.model}[/]..."):
                self._generate("Hi")

    def _generate(self, prompt: str, image: str | Path | None = None) -> str:
        """
        Generates a response using the Ollama model.

        Args:
            prompt (str): The input text prompt.
            image (str | Path | None, optional): The path to an image file. Defaults to None.

        Returns:
            str: The generated response as a string.
        """
        return ollama.generate(
            model=self.model,
            system=self.__system_prompt__,
            images=[str(image)] if image else None,
            prompt=prompt,
        ).response

    def predict(self, image: str | Path) -> Prediction:
        """
        Predicts the gender of a subject based on given image.

        Args:
            image (str | Path): The path to the image file.
        Returns:
            Prediction: A dictionary containing the predicted gender, confidence score, and reason.
        """
        response = self._generate(
            prompt="Predict the gender from the image provided.", image=image
        )
        result: Prediction = loads(response)
        return result
