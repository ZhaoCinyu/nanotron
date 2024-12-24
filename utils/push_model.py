from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

# model_path = '/playpen/xinyu/checkpoints/pt_smollm_28000_hf'
# model_path = '/playpen/xinyu/checkpoints/pt_smollm_diff_28000_hf'
# config = LlamaConfig.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(model_path)
# tokenizer = LlamaTokenizer.from_pretrained(model_path)

# config.push_to_hub("rellabear/pt_smollm_diff_28000_hf",private=True)
# model.push_to_hub("rellabear/pt_smollm_diff_28000_hf",private=True)
# tokenizer.push_to_hub("rellabear/pt_smollm_diff_28000_hf",private=True)

# model_path = '/playpen/xinyu/checkpoints/pt_smollm_28000_hf'
# config = LlamaConfig.from_pretrained(model_path)
# model = LlamaForCausalLM.from_pretrained(model_path)
# tokenizer = LlamaTokenizer.from_pretrained(model_path)

# config.push_to_hub("rellabear/pt_smollm_28000_hf",private=True)
# model.push_to_hub("rellabear/pt_smollm_28000_hf",private=True)
# tokenizer.push_to_hub("rellabear/pt_smollm_28000_hf",private=True)

model_path = '/playpen/xinyu/checkpoints/sft_smollm_base'
config = LlamaConfig.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path)

config.push_to_hub("rellabear/sft_smollm",private=True)
model.push_to_hub("rellabear/sft_smollm",private=True)
tokenizer.push_to_hub("rellabear/sft_smollm",private=True)

model_path = '/playpen/xinyu/checkpoints/sft_smollm'
config = LlamaConfig.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path)

config.push_to_hub("rellabear/sft_smollm_diff",private=True)
model.push_to_hub("rellabear/sft_smollm_diff",private=True)
tokenizer.push_to_hub("rellabear/sft_smollm_diff",private=True)
