from vllm import LLM
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

#loading vllm
def load_vllm(model, dtype, gpu_usage, max_model_len = 4096):
  llm = LLM(model=model,
            dtype=dtype,
            seed=23,
            gpu_memory_utilization=gpu_usage,
            max_model_len=max_model_len)
  return llm



def load_prm(model_id, dtype=torch.float16):
  tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
  config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
  config.pad_token_id = tokenizer.pad_token_id

  model = AutoModel.from_pretrained(
      model_id,
      config=config,
      trust_remote_code=True,
      device_map="auto",
      dtype=dtype,
  )
  return tokenizer, model