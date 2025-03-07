# Adversarial-Testing-of-Bias

Repository containing code and data for experiments on adversarial testing of bias in language models.

## Repository Structure

### Run Models
This folder contains Python scripts for running the various models used in the study:
- GPT models
- Llama models
- Deepseek models

**Requirements:**
- OpenAI API access token (for GPT models)
- Huggingface token (for Llama and Deepseek models)

To run the models, specify the model path from Huggingface and execute the code. Example configurations used in the study are provided.

### FineTuning
Contains code for fine-tuning models through the Unsloth AI platform.

The dataset for fine-tuning is dynamically generated in the `finetuning-datasets` folder.

To save your fine-tuned model to Huggingface Hub:
```python
model.push_to_hub("path", token="hf_token")
tokenizer.push_to_hub("path", token="hf_token")
```

You can then run the fine-tuned model by updating the path name in the run file of the specific model using Unsloth AI.

### Test
Testing is conducted in two ways:

1. **Bias Testing**: Using biased questions from the `sample_dataset_biased_questions` folder
2. **Standard Testing**: Using unbiased questions via `test_normal_questions.py`

The standard questions are sourced from established benchmarks:
- [Google BoolQ](https://huggingface.co/datasets/google/boolq)
- [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc)
- [WebQuestions QA Benchmarking](https://github.com/brmson/dataset-factoid-webquestions)

### Results
This folder contains all outputs from model runs and their corresponding answers.

### MRs (Metamorphic Relations)
This folder contains the process for creating questions and applying Metamorphic Relations:
- `question_generator.py`: Creates questions and applies MRs using data in the `data` folder
- `datacreator.py`: Generates sentences for MR1

## Getting Started

1. Clone the repository
2. Set up your API tokens for OpenAI and Huggingface
3. Follow the examples in the Run Models folder to execute experiments
4. Use the Test folder scripts to evaluate model performance
