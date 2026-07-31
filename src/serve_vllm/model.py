import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from vllm import LLM
from vllm.config import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

def load_vllm(model, dtype, gpu_usage, max_model_len=3072):
  llm = LLM(model=model,
            dtype=dtype,
            seed=23,
            gpu_memory_utilization=gpu_usage,
            max_model_len=max_model_len,
            enforce_eager=True,
            attention_config=AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN))
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