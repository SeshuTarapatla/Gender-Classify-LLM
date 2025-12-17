# Gender-Classify-LLM

A lightweight Python wrapper library that predicts the gender of a person in an image using local Ollama Vision-Language models.

## Requirements

- **Ollama** must be installed and running locally. You can download it from: https://ollama.com/
- Ensure the Ollama server is accessible at `http://localhost:11434` by default (port may vary).

## Features

- Uses **Qwen3-VL:4b-instruct** as the default model (you can swap it for others).
- Runs entirely locally — no internet required.
- Easy to integrate into your projects.
- Built for quick prototyping and experimentation with vision-language models.
- **Recommendation**: Use `instruct` models (e.g., `qwen3-vl:4b-instruct`, `llama3-vl:8b-instruct`) for faster inference. `thinking` models (e.g., `qwen3-vl:4b-thinking`) are also available if you need more detailed reasoning — they may be slower.

## Installation

```bash
pip install gender-classify-llm
```

## Usage

```python
from gender_classify_llm import GenderClassifier

# Default model: qwen3-vl:4b-instruct
classifier = GenderClassifier()

# Predict gender from an image
result = classifier.predict("path/to/image.jpg")
print(result)
```

## Supported Models

- `qwen3-vl:4b-instruct` (default)
- `llama3-vl:8b-instruct`
- `mistral-vl:7b-instruct`
- `qwen3-vl:4b-thinking`
- `llama3-vl:8b-thinking`
- ... (add more as needed)

## Contributing

Pull requests are welcome! Please ensure the code follows the same style and structure.

## License

MIT License — feel free to use and modify as needed.