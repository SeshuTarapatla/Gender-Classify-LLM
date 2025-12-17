__version__ = "0.1.0"

from json import loads
from pathlib import Path
from typing import Literal, TypedDict

import ollama


# Defining the system prompt for the gender classification task
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
    Represents the prediction result from the gender classification request.

    Attributes:
        gender (Literal["male", "female"]): The predicted gender of the input.
        confidence (float): The confidence level of the prediction, a value between 0 and 1.
        reason (str): A textual explanation or reasoning for the prediction.
    """

    gender: Literal["male", "female"]
    confidence: float
    reason: str


class GenderClassifier:
    """
    A classifier for determining gender from an image using local VLM.
    """

    def __init__(self, model: str = "qwen3-vl:4b-instruct") -> None:
        """
        Initialize the GenderClassifyLLM model with a specified model name.
        Args:
            model (str, optional): The name of the model to use. Defaults to "**qwen3-vl:4b-instruct**".
        """

        self.model = model
        self.__system_prompt__ = SYSTEM_PROMPT

    def predict(self, image: Path | str) -> Prediction:
        """Predict the gender of a person in an image using an LLM model.
        Args:
            image (Path | str): Path to the image file or image string.
        Returns:
            Prediction: A Prediction object containing the predicted gender, confidence, and reason.
        """
        response = ollama.generate(
            model=self.model,
            system=self.__system_prompt__,
            images=[str(image)],
            prompt="Predict the gender from the image provided.",
        ).response
        result: Prediction = loads(response)
        return result
