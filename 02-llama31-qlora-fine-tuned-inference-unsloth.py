from transformers import TextIteratorStreamer
from unsloth import FastLanguageModel
from datasets import load_dataset
from threading import Thread

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

# example question
question = 'Il prodotto PRODUCT_CODE è ordinabile? Quale è il suo stato?'

# example information/context
information = {'disponibilità': {'codice prodotto': 'PRODUCT_CODE', 'descrizione prodotto': 'PRODUCT_DESC', 'disponibile': False, 'ordinabile': False, 'stato prodotto': 'PRODUCT_STATUS'}}
# ----------------------------------------------------------------------------------------

# load quantized model with lora adapters
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=output_path_adapters,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# define a streamer for streaming output from model
streamer = TextIteratorStreamer(tokenizer)

# enable native 2x faster inference with unsloth
FastLanguageModel.for_inference(model)

# load dataset
dataset = load_dataset(
    'parquet',
    data_files={
        'train': f'data/chat-samples-en-it.parquet',
    }
)

# create a list of messages
messages = [
    {'role': 'system', 'content': 'you are a helpful assistant'},
    {'role': 'user', 'content': question},
    {'role': 'ipython', 'content': information},
]
messages = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')

# get model inputs
inputs = messages.to('cuda')

# run the generation in a separate thread, so that we can fetch the generated text in a non-blocking way
# quite ugly but.. that's how it works now with huggingface transformers :(
generation_kwargs = {'inputs': inputs, 'streamer': streamer, 'max_new_tokens': 256}
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()
generated_text = ''
for new_text in streamer:
    print(new_text, end='')
