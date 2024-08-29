# Transformers LLM LLaMA 3.1 Fine-Tuning with QLoRA 🦙🚀

Hi there! Welcome to the **transformers-llm-llama3.1-fine-tuning-qlora** repo. This project is here to help if you're interested in fine-tuning large language models like LLaMA 3.1 on consumer hardware. The goal is to share some simple and practical examples based on what I've learned while building a custom chatbot. Hopefully, this will make your fine-tuning journey a bit easier.

## Why This Repo?

Fine-tuning large language models can be a bit challenging, especially on consumer hardware. This repo is an attempt to simplify that process by sharing some sample code that worked for me. My experience comes from creating a custom chatbot, where I trained a model to generate responses. I hope these examples will be useful to you too.

## Repository Structure

Here’s what you’ll find inside:

- **/checkpoints/**: This is where the model checkpoints are stored during training.
- **/data/**: A small sample dataset for training is included here, with just 3 rows to show the required structure.
- **/models/**: Fine-tuned models will be saved in this folder.
- **requirements.txt**: All the dependencies are listed here.

## Scripts

- **01-llama31-qlora-fine-tuning.py**: The main script for fine-tuning a quantized version of LLaMA 3.1-8B using LoRA adapters. You can choose to save either just the adapters or the whole model in GGUF format.
- **02-llama31-qlora-fine-tuned-inference-unsloth.py**: A script for running inference with a fine-tuned model using Hugging Face Transformers and Unsloth.
- **03-llama31-qlora-fine-tuned-inference-llamacpp.py**: A script for running inference with a fine-tuned model in GGUF format using LLaMA.cpp.

## Data Structure Note

LLaMA 3.1 introduces a new role in IPython that allows you to include context and information in a JSON-style format. This can be really handy when building chatbots since it can store the output from Python tools. Keep this in mind as you set up your data! In my examples you'll notice I'm using this role, in case you don't need.. slightly adjust the code removing it!

## Requirements and Installation

Before getting started, you’ll need to install some dependencies. Here’s how I suggest to do it:

```bash
python -m pip install --upgrade pip wheel setuptools
pip install ninja==1.11.1.1
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git@637ed8c6bd252f981e89e30e1085efc03a06a880"

# optional (needed if you want to work with llama.cpp and models in gguf format)
pip install llama_cpp_python==0.2.87
```

Otherwise you can try to directly install the **`requirements.txt`** file.

I've tested everything with Python 3.11.9 on WSL (standard Ubuntu distro).

Keep in mind that the Unsloth runtime works natively only on Linux, you can't directly use it on Windows, but WSL it's a quick and effective alternative.

## Have Fun!

And that's it! You're ready to start fine-tuning your own LLaMA 3.1 models. Enjoy exploring and happy fine-tuning! 😄
