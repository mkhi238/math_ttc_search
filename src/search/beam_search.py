
#IMPORTS
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from eval.load_and_parse import make_math_parser, load_dataset
from serve_vllm.model import load_vllm, load_prm
from vllm import SamplingParams
import pandas as pd
import heapq
from scipy.special import softmax
import numpy as np
import random
import time
from math_verify import parse, verify
from difflib import SequenceMatcher
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


#PARAMETERS
N = [1, 2, 4, 8, 16]
M = 4
TEMPRATURE = 0.4
MAX_TOKENS = 200
CHECKPOINT = 50
STOP_WORDS = ["\n\n", "Step"]
MAX_ROUNDS = 10
ALPHA = 0.65
STOCHASTIC_TEMP = 5.0
SIZE = 3

#PROMPT INTRO
INTRO = INTRO = r"Solve the following math problem efficiently and clearly, using a step-by-step format. Conclude with the final answer in the form $\boxed{answer}$."

#FEW SHOT EXAMPLES
FEW_SHOT_EXAMPLES = [
"""Problem: Simplify $\\frac{3}{4} + \\frac{1}{6}$. Express your answer as a common fraction.
Step 1: Find a common denominator: $\\frac{3}{4} = \\frac{9}{12}$ and $\\frac{1}{6} = \\frac{2}{12}$.
Step 2: Add the fractions: $\\frac{9}{12} + \\frac{2}{12} = \\frac{11}{12}$.
The final answer is $\\boxed{\\frac{11}{12}}$.""",

"""Problem: A right triangle has legs of length $6$ and $8$. What is the length of the hypotenuse?
Step 1: Apply the Pythagorean theorem: $c^2 = 6^2 + 8^2$.
Step 2: Compute: $c^2 = 36 + 64 = 100$.
Step 3: Take the square root: $c = \\sqrt{100} = 10$.
The final answer is $\\boxed{10}$.""",

"""Problem: Evaluate $\\sin\\left(\\frac{\\pi}{6}\\right) + \\cos\\left(\\frac{\\pi}{3}\\right)$.
Step 1: Recall $\\sin\\left(\\frac{\\pi}{6}\\right) = \\frac{1}{2}$.
Step 2: Recall $\\cos\\left(\\frac{\\pi}{3}\\right) = \\frac{1}{2}$.
Step 3: Add: $\\frac{1}{2} + \\frac{1}{2} = 1$.
The final answer is $\\boxed{1}$.""",

"""Problem: How many positive divisors does $60$ have?
Step 1: Prime factorize: $60 = 2^2 \\cdot 3^1 \\cdot 5^1$.
Step 2: Use the divisor-count formula: $(2+1)(1+1)(1+1) = 3 \\cdot 2 \\cdot 2$.
Step 3: Multiply: $3 \\cdot 2 \\cdot 2 = 12$.
The final answer is $\\boxed{12}$.""",

"""Problem: Solve for $x$: $2x + 5 = 17$.
Step 1: Subtract $5$ from both sides: $2x = 12$.
Step 2: Divide both sides by $2$: $x = 6$.
The final answer is $\\boxed{6}$.""",

"""Problem: Simplify $\\sqrt{72}$.
Step 1: Factor out the largest perfect square: $72 = 36 \\cdot 2$.
Step 2: Take the square root: $\\sqrt{36 \\cdot 2} = 6\\sqrt{2}$.
The final answer is $\\boxed{6\\sqrt{2}}$.""",

"""Problem: Two fair six-sided dice are rolled. What is the probability that the sum of the two dice is $7$?
Step 1: There are $6 \\times 6 = 36$ total outcomes.
Step 2: The outcomes summing to $7$ are $(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)$, giving $6$ favorable outcomes.
Step 3: The probability is $\\frac{6}{36} = \\frac{1}{6}$.
The final answer is $\\boxed{\\frac{1}{6}}$.""",

"""Problem: Find the larger root of $x^2 - 5x + 6 = 0$.
Step 1: Factor: $x^2 - 5x + 6 = (x-2)(x-3)$.
Step 2: The roots are $x = 2$ and $x = 3$.
Step 3: The larger root is $3$.
The final answer is $\\boxed{3}$.""",

"""Problem: Compute $|3 + 4i|$, the modulus of the complex number $3+4i$.
Step 1: The modulus is $\\sqrt{a^2+b^2}$ where $a=3$ and $b=4$.
Step 2: Compute: $\\sqrt{3^2+4^2} = \\sqrt{9+16} = \\sqrt{25} = 5$.
The final answer is $\\boxed{5}$.""",

"""Problem: Find the sum of the infinite geometric series $1 + \\frac{1}{2} + \\frac{1}{4} + \\frac{1}{8} + \\cdots$.
Step 1: The series has first term $a=1$ and common ratio $r=\\frac{1}{2}$.
Step 2: The sum is $\\frac{a}{1-r} = \\frac{1}{1-\\frac{1}{2}} = 2$.
The final answer is $\\boxed{2}$.""",

"""Problem: Evaluate $\\log_3 81$.
Step 1: Write $81$ as a power of $3$: $81 = 3^4$.
Step 2: So $\\log_3 81 = \\log_3 3^4 = 4$.
The final answer is $\\boxed{4}$.""",

"""Problem: Write $\\frac{3}{8}$ as a decimal.
Step 1: Divide $3$ by $8$: $3 \\div 8 = 0.375$.
The final answer is $\\boxed{0.375}$.""",

"""Problem: How many ways can a committee of $2$ people be chosen from a group of $6$ people?
Step 1: The number of ways is $\\binom{6}{2}$.
Step 2: Compute: $\\binom{6}{2} = \\frac{6 \\cdot 5}{2 \\cdot 1} = 15$.
The final answer is $\\boxed{15}$.""",

"""Problem: Find the midpoint of the segment connecting the points $(2,3)$ and $(6,7)$.
Step 1: The midpoint formula is $\\left( \\frac{x_1+x_2}{2}, \\frac{y_1+y_2}{2} \\right)$.
Step 2: Compute: $\\left( \\frac{2+6}{2}, \\frac{3+7}{2} \\right) = (4, 5)$.
The final answer is $\\boxed{(4, 5)}$.""",

"""Problem: Evaluate $\\cos(60^\\circ)$.
Step 1: Recall the standard value $\\cos(60^\\circ) = \\frac{1}{2}$.
The final answer is $\\boxed{\\frac{1}{2}}$.""",
]

def text_similarity(a, b):
  return SequenceMatcher(isjunk = None, a = a, b = b, autojunk=False).ratio()

def create_sampling_parameters(n, temperature=TEMPRATURE, max_tokens=MAX_TOKENS, stop = STOP_WORDS):
    sampling_parameters = SamplingParams(n = n, temperature=temperature, max_tokens=max_tokens, stop = stop, logprobs = 1)
    return sampling_parameters
  
def format_beam_prompt(question, beam_so_far, step_num, pool = FEW_SHOT_EXAMPLES, k = 5):
  items = random.sample(pool, k)
  example_txt = "\n\n".join(items)

  return (
    INTRO + "\n\n" +
    example_txt + 
    f"\n\nProblem: {question}\n" +
    beam_so_far +
    f"Step {step_num}:"
  )
  
def format_prm_prompt(question, text_so_far):
    steps = text_so_far.strip().split("\n\n")
    return question + "\n\n" + "<extra_0>".join(steps) + "<extra_0>"

def length_norm(seq_len, alpha = ALPHA):
  return (5+seq_len)**alpha / 6**alpha


def prm_scoring(question, candidates, prm_tokenizer, prm_model):
  
  step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
  out = []
  for c in candidates:
    prompts = format_prm_prompt(question, c['text_so_far'])
    #input_ids [15, 892, 33, 4021, step_sep_id, 77, 12, step_sep_id, ...]
    input_ids = prm_tokenizer.encode(prompts, return_tensors="pt").to(prm_model.device) #(1,N) from return_tensor = 'pt'
    with torch.no_grad():
      output = prm_model(input_ids=input_ids)
    logits = output.logits #(batch = 1, seq len = N, output class (T,F) = 2)
    #token masks [False, False, False, False, True, False, False, True, ...]
    token_masks = (input_ids == step_sep_id) #(1,N)
    #probs is (1,N,2)
    probs = F.softmax(logits, dim=-1) #(1,N,2)
    masked_probs = probs * token_masks.unsqueeze(2)
    
    nonzero_probs = masked_probs[0][masked_probs[0] != 0] #masked_probs[0] pulls out batch dim, leaving (N,2)
    step_rewards = nonzero_probs.view(-1, 2)[:, 1]
    
    if len(step_rewards) > 0:
      score = step_rewards[-1].item()
    else:
      score = 0.0
      
    out.append(score)
  return out

def beam_search(question, llm, prm_tokenizer, prm_model, n, m, method = "standard"):
  beams = [{"text_so_far": "", "score": 0.0, "step": 1, "num_tokens": 0, "finished": False}]
  round = 0

  while round < MAX_ROUNDS:
    candidates = []
    total_finished = sum(item['finished'] for item in beams)
    if total_finished == len(beams):
      break
    
    for b in beams:
      if b["finished"] == True:
        candidates.append(b)
        continue
      
      else:
        #build one prompt per active beam
        prompt = []
        prompt.append(format_beam_prompt(question, b['text_so_far'], b['step']))
        n_this_round = n if round == 0 else m
        samples = create_sampling_parameters(n = n_this_round)
        
        output = llm.generate(prompt, samples)
        
        for completion in output[0].outputs:
          candidates.append({
          "text_so_far": b['text_so_far'] + "\n\n" + f"Step {b['step']}:" + completion.text,
          "score": b['score'] + completion.cumulative_logprob,
          "step": b['step'] + 1,
          "num_tokens": b["num_tokens"] + len(completion.token_ids),
          "finished": "\\boxed{" in completion.text,
          })
          
    if method == "standard":    
      beams = heapq.nlargest(n, candidates, key = lambda x: x['score'] / length_norm(x['num_tokens']))
      
    elif method == "probabalistic":
      beams = []
      scores = np.array([c['score'] / length_norm(c['num_tokens']) for c in candidates])
      probs = softmax(scores / STOCHASTIC_TEMP)
      idx = np.random.choice(len(candidates), size = min(n, len(candidates)), replace = False, p = probs)
      for i in idx:
        beams.append(candidates[i])
        
    elif method == "PRM":
      beams = []
      scores = prm_scoring(question, candidates, prm_tokenizer, prm_model)
      top_idx = heapq.nlargest(n, range(len(candidates)), lambda i : scores[i])
      for i in top_idx:
        beams.append(candidates[i])
    
    #Extra beam search method, Diverse Beam Search
    elif method == "diverse":
      num_groups = min(2,n)
      diversity_penalty = 0.5
      group_size =n // num_groups

      def dbs_score(c):
        base = c['score'] / length_norm(c['num_tokens'])
        overlap_score = 0
        
        for prev in beams:
          overlap_score += text_similarity(c['text_so_far'], prev['text_so_far'])
        return base - diversity_penalty * overlap_score
      
      beams = []
      remaining = candidates.copy()
      
      for _ in range(num_groups):
        group_pick = heapq.nlargest(group_size, remaining, key=dbs_score)
        beams.extend(group_pick)
        remaining = [c for c in remaining if c not in group_pick] 
          
          
    round += 1
  
  finished = []     
  for b in beams:
    if b['finished']:
      finished.append(b)
  if not finished:
    return None
  best = heapq.nlargest(1, finished, key = lambda x: x['score'] / length_norm(x['num_tokens']))
  return best[0]['text_so_far']

def make_parse(resp):
  if resp is None:
    return None
  return parse(resp)

def check_correct(row):
  if row['y_pred'] is None:
    return 0
  try:
    return int(verify(row['y_true'], row['y_pred']))
  except Exception:
    return 0

if __name__ == "__main__":
  #Monkey Patching to add DynamicCache.from_legacy_cache needed for PRM model from prev HF transformers versions
  #Need to Monkey Patch 4 different potential throws
  if not hasattr(DynamicCache, "from_legacy_cache"):
    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
      cache = cls()
      if past_key_values is not None:
        if isinstance(past_key_values, cls):
          return past_key_values
        for layer_idx in range(len(past_key_values)):
          key_states, value_states = past_key_values[layer_idx]
          cache.update(key_states, value_states, layer_idx)
      return cache
    DynamicCache.from_legacy_cache = from_legacy_cache
    
  if not hasattr(DynamicCache, "get_usable_length"):
    def get_usable_length(self, new_seq_length, layer_idx=0):
      previous_seq_length = self.get_seq_length(layer_idx)
      return previous_seq_length
    DynamicCache.get_usable_length = get_usable_length
    
  if not hasattr(DynamicCache, "get_max_length"):
    def get_max_length(self):
      return None
    DynamicCache.get_max_length = get_max_length
    
  if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
  
  #LOAD VLLM
  llm = load_vllm(f"Qwen/Qwen2.5-{SIZE}B-Instruct", dtype='float16', gpu_usage=0.5)
  print('loaded llm')
  prm_tokenizer, prm_model = load_prm("Qwen/Qwen2.5-Math-PRM-7B")
  print('loaded prm')
  
  #LOAD DATA
  ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
  df = pd.DataFrame(ds)
  df = make_math_parser(df, 'answer', 'parsed_answer')
  
  #GENERATE & EXTRACT SAMPLES - STANDARD
  for iter in N:
    results = {}
    for idx, q in enumerate(df['problem']):
      start = time.time()
      resp = beam_search(q, llm, prm_tokenizer, prm_model,n = iter, m = M, method = "standard")
      elapsed = time.time() - start
      results[q] = (make_parse(resp), elapsed)
      if idx % CHECKPOINT == 0:
        results_df = pd.DataFrame(
        [(q, a, t) for q, (a, t) in results.items()],
        columns=['problem', 'y_pred', 'time_seconds'])
        results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
        results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
        results_df['correct'] = results_df.apply(check_correct, axis=1)
        results_df.to_csv(f'results/beam_search_standard_MATH_3B_{iter}_beam(s).csv', index=False)
        
    #COLLECT RESULTS - STANDARD
    results_df = pd.DataFrame(
    [(q, a, t) for q, (a, t) in results.items()],
    columns=['problem', 'y_pred', 'time_seconds'])
    results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
    results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
    results_df['correct'] = results_df.apply(check_correct, axis=1)
    results_df.to_csv(f'results/beam_search_standard_MATH_3B_{iter}_beam(s)', index=False)
    
  #GENERATE & EXTRACT SAMPLES - PROBABALISTIC
  for iter in N:
    results = {}
    for idx, q in enumerate(df['problem']):
      start = time.time()
      resp = beam_search(q, llm, prm_tokenizer, prm_model, n = iter, m = M, method = "probabalistic")
      elapsed = time.time() - start
      results[q] = (make_parse(resp), elapsed)
      if idx % CHECKPOINT == 0:
        results_df = pd.DataFrame(
        [(q, a, t) for q, (a, t) in results.items()],
        columns=['problem', 'y_pred', 'time_seconds'])
        results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
        results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
        results_df['correct'] = results_df.apply(check_correct, axis=1)
        results_df.to_csv(f'results/beam_search_prob_MATH_3B_{iter}_beam(s)', index=False)
    
    #COLLECT RESULTS - PROBABALISTIC 
    results_df = pd.DataFrame(
    [(q, a, t) for q, (a, t) in results.items()],
    columns=['problem', 'y_pred', 'time_seconds'])
    results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
    results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
    results_df['correct'] = results_df.apply(check_correct, axis=1)
    results_df.to_csv(f'results/beam_search_prob_MATH_3B_{iter}_beam(s)', index=False)
  
  
  # #GENERATE & EXTRACT SAMPLES - DBS
  for iter in N:
    results = {}
    for idx, q in enumerate(df['problem']):
      start = time.time()
      resp = beam_search(q, llm, prm_tokenizer, prm_model, n = iter, m = M, method = "diverse")
      elapsed = time.time() - start
      results[q] = (make_parse(resp), elapsed)
      if idx % CHECKPOINT == 0:
        results_df = pd.DataFrame(
        [(q, a, t) for q, (a, t) in results.items()],
        columns=['problem', 'y_pred', 'time_seconds'])
        results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
        results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
        results_df['correct'] = results_df.apply(check_correct, axis=1)
        results_df.to_csv(f'results/beam_search_diverse_MATH_3B_{iter}_beam(s)', index=False)
        
    #COLLECT RESULTS - DBS
    results_df = pd.DataFrame(
    [(q, a, t) for q, (a, t) in results.items()],
    columns=['problem', 'y_pred', 'time_seconds'])
    results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
    results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
    results_df['correct'] = results_df.apply(check_correct, axis=1)
    results_df.to_csv(f'results/beam_search_diverse_MATH_3B_{iter}_beam(s)', index=False)
  
  #GENERATE & EXTRACT SAMPLES - VALUE GUIDED
  for iter in N:
    results = {}
    for idx, q in enumerate(df['problem']):
      start = time.time()
      resp = beam_search(q, llm, prm_tokenizer, prm_model, n = iter, m = M, method = "PRM")
      elapsed = time.time() - start
      results[q] = (make_parse(resp), elapsed)
      if idx % CHECKPOINT == 0:
        results_df = pd.DataFrame(
        [(q, a, t) for q, (a, t) in results.items()],
        columns=['problem', 'y_pred', 'time_seconds'])
        results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
        results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
        results_df['correct'] = results_df.apply(check_correct, axis=1)
        results_df.to_csv(f'results/beam_search_value_guided_MATH_3B_{iter}_beam(s)', index=False)
    
    #COLLECT RESULTS - VALUE GUIDED 
    results_df = pd.DataFrame(
    [(q, a, t) for q, (a, t) in results.items()],
    columns=['problem', 'y_pred', 'time_seconds'])
    results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
    results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
    results_df['correct'] = results_df.apply(check_correct, axis=1)
    results_df.to_csv(f'results/beam_search_value_guided_MATH_3B_{iter}_beam(s)', index=False)
    

