# Adversarial-Testing-of-Bias

Repository containing code and data for experiments on adversarial testing of bias in language models.

The growing adoption of Large Language Models (LLMs) in various applications has raised concerns about biases in their outputs. Recent studies have attempted to assess bias in LLMs by posing direct questions related to protected attributes (e.g., "Are men meaner than women?"). However, real-world scenarios often involve more nuanced or indirect questions that may still induce bias, such as those with added noise or those that reference protected attributes subtly (e.g., "Is being mean a common trait among some men?"). In this research, we leverage concepts from adversarial machine learning testing to investigate bias in LLM responses. We propose five Metamorphic Relations (MRs) designed to automatically modify questions based on two types of bias metadata: protected groups and their associated attributes. Using these MRs, we rephrase questions (e.g., by adding contextual details about groups or attributes) while expecting consistent, unbiased responses across both original and mutated versions.

To validate our approach, we conducted experiments using a dataset comprising known bias-triggering questions from existing literature, alongside an evaluation method provided by the same sources. We applied our methodology to six LLMs: Llama 3.1-8B-Instruct, Llama 3.1-70B-Instruct, Llama 3.2-3B-Instruct, DeepSeek-R1-Distill-Llama-8B, GPT-3.5-Turbo, and GPT-4o-Mini. Our results demonstrate that the proposed Metamorphic Relations (MRs) uncover approximately twice as many biases in the benchmark dataset compared to BiasAsker, a state-of-the-art black-box LLM bias testing tool that also supplied the dataset and evaluation method. Additionally, we fine-tuned the LLMs using our MRs, significantly improving their resilience to producing biased responses. Specifically, Llama 3.1-8B-Instruct became 1.93 times more resilient, Llama 3.2-3B-Instruct 1.37 times, and DeepSeek-R1-Distill-Llama-8B 1.05 times, all without compromising their general performance.

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
