from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer, LlamaForCausalLM
import torch


# checkpoint = "/playpen/xinyu/checkpoints/sft_smollm"
checkpoint = '/playpen/xinyu/checkpoints/pt_smollm_28000_hf'
tknzer = 'HuggingFaceTB/SmolLM-135M'

device = "cuda" # for GPU usage or "cpu" for CPU usage
config = AutoConfig.from_pretrained(checkpoint)
tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = LlamaForCausalLM.from_pretrained(checkpoint, 
    torch_dtype=torch.float16, device_map="cuda", 
    # attn_implementation='differential').to(device)
    attn_implementation='flash_attention_2').to(device)

# messages = [{"role": "user", "content": "What do you think the future of AI?"}]
# input_text=tokenizer.apply_chat_template(messages, tokenize=False)
input_text='The future of AI is'
print(input_text)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
