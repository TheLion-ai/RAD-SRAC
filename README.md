# Simple Retrieval Augmented Classification for Radiology
## Features

- Support for multiple medical imaging datasets (brain tumor, coronahack, kits23)
- Integration with various LLM models for image analysis
- Retrieval-Augmented Generation (RAG) for improved analysis
- Command-line interface for easy use
- Jupyter notebooks for interactive analysis and metric calculation

## Setup
### Qdrant
Start Qdrant using docker
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```
### LLMS
Configure LLMS using `src/config/models.py`

We are using litellms that supports most LLM providers using cloud APIs as well as local setups using e.g. ollama and LLM studio.

For testing various LLMS we recommend OpenRouter

Set up your API keys in `.env` file
e.g
```
ANTHROPIC_API_KEY = ...
OPENAI_API_KEY = ...
OPENROUTER_API_KEY = ...
```
### Encoder
For the encoder we are using MedImageInsight implementation provided by lion-ai - https://huggingface.co/lion-ai/MedImageInsights
The repo shows you how to set up FastAPI service using the model
You can then add the url to your service in `.env` file
```
MEDIMAGEINSIGHT_URL = ...
```
If you want to use custom encoders simply implemet a class that has the `encode()` method (see `src/encoders/medimageinsight.py`) and add it to `src/config/encoder.py`
```
encoders = {
    "YourEncoderName": YourCustomEncoderClass
}
```
### Datasets

For datasets we are using UMIE-datasets https://huggingface.co/datasets/lion-ai/umie_datasets

Download the dataset you want to use and put its folder in `/data` dir
e.g
```
data/
└── 00_kits23/
    ├── CT/
    │   ├── Images/
    │   └── Masks/
    ├── 00_kits23_one_img_per_study.jsonl
    └── 00_kits23.jsonl

```
Next configure dataset in `src/config/datasets.py`
e.g.
```
    "kits23": Dataset(
        name="00_kits23",
        modality="CT",
        classes=[
            "angiomyolipoma",
            "chromophobe_rcc",
            "clear_cell_rcc",
            "oncocytoma",
            "papillary_rcc",
        ],
        val_file = "splits/00_kits23/val.jsonl",
        train_file= "splits/00_kits23/train.jsonl"
    ),
```
You can use custom train/test splits using `val_file` and `train_file`

We have privided the splts we used for our experimets in the
`splits` directory
### Requirements
Install uv package manager https://docs.astral.sh/uv/

then run:
```
uv sync
```
## Running the experimets
### Vectorize the dataset
```
uv run cli.py upload_dataset
```
The script will prompt you to select dataset and encoder

### Run predictions
```
uv run cli.py run
```
CLI Options for the run Command
When using the run command in the CLI, you have the following options:

`--dataset`: Select the dataset to use (required)

* Choices are based on the datasets configured in src/config/datasets.py

`--model`: Select the model to use for analysis (required)

* Choices are based on the models configured in src/config/models.py

`--encoder`: Select the encoder to use (required)

* Choices are based on the encoders configured in src/config/encoder.py

`--few-shot/--no-few-shot`: Enable or disable few-shot learning (default: False)

`--n-images`: Number of relevant images to show (default: 5)

`--classes/--no-classes`: Include or exclude true classes in the few-shot prompt (default: True)

`--roco:` Use ROCO database instead of the dataset database (flag)

`--verbose`: Enable verbose logging for each prediction (flag)

Example usage:

```
python cli.py run --dataset "kits23" --model "Qwen2-VL 7B Instruct" --encoder MedImageInsight  --few-shot --classes
```
## Analyze experiments
After each experiment the CLI will print metrics for the experiment.

Running each experiment will also create `.jsonl` file in `results` dir 


Use `src/calc_metrics.ipynb` if you want to analyze past experiments
