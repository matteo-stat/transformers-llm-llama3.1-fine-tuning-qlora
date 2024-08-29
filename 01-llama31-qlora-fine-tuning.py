from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ----------------------------------------------------------------------------------------
# max sequence length that the model will receive as input
max_seq_length = 2048 

# leave to none for auto detection (otherwise can be float16 or blfloat16, depending on your gpu)
dtype = None

# use 4 bit quantization to reduce memory usage
load_in_4bit = True

# seed for reproducibility
seed = 1993

# output path for lora adapters
output_path_adapters = 'models/llama31-8b-qlora-adapters-only'
output_path_gguf = 'models/llama31-8b-qlora-gguf'
# ----------------------------------------------------------------------------------------

# load dataset
dataset = load_dataset(
    'parquet',
    data_files={
        'train': f'data/chat-samples-en-it.parquet',
    }
)

# load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# add lora adapters to the model, below some notes
# --- r (lora matrix rank) ---
# higher ranks can store more information but increase computational load and memory cost
# typical rank values are between 8 and 256 (suggested value 8, 16, 32, 64, 128)
# try to start with lower values and then increase if needed
# --- lora_alpha (alpha) ---
# scaling factor for updates that impact the lora adapters contribution to the model output
# typical values are 1x or 2x the rank value
# --- target modules ---
# lora can be applied to various model layers
# if more layers are choosen then memory cost increase as the number of trainable parameters
model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=16,
    lora_dropout=0, # 0 is optimized with unsloth
    bias='none',    # 'none' is optimized with unsloth
    use_gradient_checkpointing='unsloth', # use 'unsloth' to reduce vram usage
    random_state=seed,
    max_seq_length=max_seq_length,
    use_rslora=False,  # rank stabilized lora
)

# training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=1,
    optim='adamw_8bit',
    weight_decay=0.01,
    lr_scheduler_type='linear',
    seed=seed,
    output_dir='checkpoints/llama31-8b-qlora-fine-tuned',
)

# function for apply a chat template to conversations in the dataset
# "information" column contains a json providing context about question for helping the llm generate a proper answer
# if you don't have the "information" column simply remove it from the function below
def apply_chat_template(examples):
    chat_samples = []
    for question, information, answer in zip(examples['question'], examples['information'], examples['answer']):
        chat = [
            {'role': 'system', 'content': 'you are a helpful assistant'},
            {'role': 'user', 'content': question},
            {'role': 'ipython', 'content': information},
            {'role': 'assistant', 'content': answer}
        ]
        chat_sample = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        chat_samples.append(chat_sample)

    return {'text': chat_samples}

# apply a chat template to conversations in the dataset
dataset = dataset.map(apply_chat_template, batched=True)

# trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    dataset_text_field='text',
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False, # can make training 5x faster for short sequences
    args=training_args,
)

# start training
trainer_stats = trainer.train()

# save only lora adapters weights
model.save_pretrained(output_path_adapters)
tokenizer.save_pretrained(output_path_adapters)

# alternative method for saving only lora adapters
# model.save_pretrained_merged(output_path_adapters, tokenizer, save_method='lora')

# save quantized versions, useful for using it in other frameworks like llama.cpp
# WARNING: this is gonna merge and quantize also the lora adapters layers, reducing their effectiveness
# be aware of this and expect worse results compared to the model with lora adapters at full precision
model.save_pretrained_gguf(output_path_gguf, tokenizer, quantization_method='q5_k_m')
