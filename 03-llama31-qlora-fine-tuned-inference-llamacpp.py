from llama_cpp import Llama

# ----------------------------------------------------------------------------------------
# max sequence length that the model will receive as input
max_seq_length = 2048 

# seed for reproducibility
seed = 1993

# output path for lora adapters
path_gguf = 'models/llama31-8b-qlora-gguf'

# example question
question = 'The product PRODUCT_CODE is it orderable?'

# example information/context
information = {'availability': {'product code': 'PRODUCT_CODE', 'product desc': 'PRODUCT_DESC', 'available': False, 'orderable': False, 'product status': 'PRODUCT_STATUS'}}
# ----------------------------------------------------------------------------------------

# load quantized model
llm = Llama(
    model_path=f'{path_gguf}/unsloth.Q5_K_M.gguf',
    n_gpu_layers=-1,
    seed=seed,
    n_ctx=max_seq_length, 
    chat_format='llama-3'
)

# create a list of messages
messages = [
    {'role': 'system', 'content': 'you are a helpful assistant'},
    {'role': 'user', 'content': question},
    {'role': 'ipython', 'content': information}
]

# format messages and prepare output from llm
output = llm.create_chat_completion(
    messages=messages,
    max_tokens=512,
    stream=True
)

# stream output from llm
for chunk in output:
    delta = chunk['choices'][0]['delta']
    if 'role' in delta:
        print(delta['role'], end=': ')
    elif 'content' in delta:
        print(delta['content'], end='')
