# Sentiment Analysis Workshop
## Installation
The project is setup by using [uv package management](https://docs.astral.sh/uv/). After installing uv, you can install packages via command
```console
uv venv
source .env/bin/activate
uv sync
```
For LLM serving on a local machine, it is using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Please refer official documents for troubleshooting. The project only support CPU running, due to being gpu-poor. Therefore, It isn't tested on GPU settings.

## Running Code
The project support basic cli inference in the root directory of the project, you can generate responses from experiments via CLI. the first run might take some time due to downloading models.

```console
uv run main.py list
```
Returns avaliable models with respective prompts

```console
uv run main.py run --model <model_name> --text <text>
```
Runs a single inference for the given text using the selected model and prompt.

## Project Structure

* **experiments:** Contains prompt and model definitions used for running experiments.
* **notebooks:** Contains Jupyter notebooks for data analysis and tracking the state of experiments.
* **docs:** Contains plots and the `report.md` file, which is used to documentation of the experiments and data related analysis.

## Adding New Prompts or Models

Prompts and models are stored within the `experiments` package. This package contains classes that extend the base class defined in `experiment.py`. To add a new prompt or model, you can extend this base class, implement your own pipeline, and then add it to `main.py`.