# selection walks down from root via UCT, lands on some node with no children. That's the "come to a node" part.

# expansion fires on that node, generates M candidate next steps, creates M new child nodes, all n=0, w=0. This is the part your description skipped slightly, it's not "find the score of that branch," it's first "create the branch," M of them at once, and only after they exist does scoring happen.

# simulation (Option 1, PRM-direct, what you're building right now) takes one of those M new children, hands its current partial state_text to the PRM, gets back one scalar, how promising this partial branch looks. That's your "find the score" step, and to be exact, it's scoring one specific child's state, not "that branch" abstractly, whichever single child you picked to evaluate this iteration.

# backprop takes that scalar and walks it up the path, root to the scored child, updating n and w at every node along the way. This is exactly your "update Q value" step, Q isn't stored directly, remember, it's computed live as w/n whenever UCT needs it, but backprop is what keeps w and n accurate so that future Q reads are correct.

# Then yes, continue, next iteration calls selection(root) again, and because the numbers just changed, UCT might send it down a different path this time.

#Expansion
#IMPORTS
M = 4
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from eval.load_and_parse import make_math_parser, load_dataset
from serve_vllm.model import load_vllm, load_prm
from vllm import SamplingParams
import pandas as pd
import heapq
import numpy as np
import random
import time
from math_verify import parse, verify
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


#PARAMETERS

TEMPRATURE = 0.4
MAX_TOKENS = 200
CHECKPOINT = 50
STOP_WORDS = ["\n\n", "Step"]
MAX_ROUNDS = 1
SIZE = 1.5
NUM_FEWSHOT_POOLS = 12

#PROMPT INTRO
INTRO = r"Solve the following math problem efficiently and clearly, using a step-by-step format. Conclude with the final answer in the form $\boxed{answer}$."

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

FEWSHOT_POOLS = ["\n\n".join(random.sample(FEW_SHOT_EXAMPLES, 5)) for _ in range(NUM_FEWSHOT_POOLS)]

def create_sampling_parameters(n, temperature=TEMPRATURE, max_tokens=MAX_TOKENS, stop = STOP_WORDS):
    sampling_parameters = SamplingParams(n = n, temperature=temperature, max_tokens=max_tokens, stop = stop, logprobs = 1)
    return sampling_parameters
  
def format_prompt(question, question_idx, beam_so_far, step_num):
  example_txt = FEWSHOT_POOLS[question_idx % NUM_FEWSHOT_POOLS]
  return (
    INTRO + "\n\n" +
    example_txt +
    f"\n\nProblem: {question}\n" +
    beam_so_far +
    f"Step {step_num}:"
  )

def format_node_estimate_prompt(question, text_so_far):
    steps = text_so_far.strip().split("\n\n")
    return question + "\n\n" + "<extra_0>".join(steps) + "<extra_0>"
  
def prm_scoring(question, question_idx, node, prm_tokenizer, prm_model):
  
  step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
  out = []

  prompts = format_node_estimate_prompt(question, node.state_text)
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
  
def UCT(node, c = 2, eps=1e-8):
  if node.n != 0:
    n = node.n
  else:
    n = eps
  if node.parent.n != 0:
    parent_n = node.parent.n
  else:
    parent_n = eps
  q = node.w / n
  uct = q + c * np.sqrt(np.log(parent_n)/n)
  return uct

class MCTSNode():
  def __init__(self, state_text, step_num, n = 0, w = 0, terminal_flag = None, parent = None):
    self.state_text = state_text
    self.step_num = step_num
    self.parent = parent
    self.terminal_flag = terminal_flag
    self.n = n #couter: how many times search passed thru this node
    self.w = w #backprop: total value backprop through the node
    self.children = []
  
  def select_best_node(self, children):
    top_child = heapq.nlargest(1, children, key= lambda child: UCT(child))[0]
    return top_child
  
def selection(node):
  curr = node
  while curr.children:
    curr = curr.select_best_node(curr.children)
  return curr

def expansion(node, question, question_idx, llm):
  if node.terminal_flag == True: #node is done expanding
    return (node, False)
  
  prompt = format_prompt(question, question_idx, node.state_text, node.step_num)
  samples = create_sampling_parameters(n = M)
  outputs = llm.generate(prompt, samples)
  
  for completion in outputs[0].outputs:
    child_state = node.state_text + completion.text
    child_step_num = node.step_num + 1
    is_terminal = "\\boxed" in completion.text or (child_step_num >= MAX_ROUNDS)
    child = MCTSNode(child_state, child_step_num, parent = node, terminal_flag=is_terminal)
    node.children.append(child)
  
  return (node.children, True)

def simulation(question, question_idx, node,  prm_tokenizer, prm_model):  
  score = prm_scoring(question, question_idx, node, prm_tokenizer, prm_model)
  return score[0]

def backpropagation(node, score):
  curr = node
  while curr is not None:
    prev = curr
    curr.n += 1
    curr.w += score
    curr = curr.parent
  return prev
  
def mcts_loop(root, num_iters, question, question_idx, llm, prm_tokenizer, prm_model):
  for _ in range(num_iters):
    leaf = selection(root)
    result, expanded = expansion(leaf, question, question_idx, llm)
    ts = result[0] if expanded else result
    score = simulation(question, question_idx, ts, prm_tokenizer, prm_model) 
    root = backpropagation(ts, score)
  
  return root

def get_final_answer(root, mode = "most_visited"):
  if mode == "most_visited":
    return max(root.children, key = lambda c: c.n)
  
  else:
    return max(root.children, key=lambda c: c.w / c.n if c.n != 0 else 0)
  
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
  llm = load_vllm(f"Qwen/Qwen2.5-{SIZE}B-Instruct", dtype='float16', gpu_usage=0.65)
  print('loaded llm')
  prm_tokenizer, prm_model = load_prm("Qwen/Qwen2.5-Math-PRM-7B")
  print('loaded prm')
  
  #LOAD DATA
  ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
  df = pd.DataFrame(ds)
  df = make_math_parser(df, 'answer', 'parsed_answer')
  
  for num_iters in [10,50.100]:
    results = {}
    for idx, q in enumerate(df['problem']):
      start_time = time.time()
      root = MCTSNode("", 0, n=0, w=0, terminal_flag=None, parent=None)
      root = mcts_loop(root, num_iters, q, idx, llm, prm_tokenizer, prm_model)
      final = get_final_answer(root, mode="most_visited")
      elapsed = time.time() - start_time
      results[q] = (make_parse(final.state_text), elapsed)
      if idx % CHECKPOINT == 0:
        results_df = pd.DataFrame(
        [(q, a, t) for q, (a, t) in results.items()],
        columns=['problem', 'y_pred', 'time_seconds'])
        results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
        results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
        results_df['correct'] = results_df.apply(check_correct, axis=1)
        results_df.to_csv(f'/mnt/d/math_ttc_search/results/mcts_MATH_1.5B_{num_iters}_beam(s).csv', index=False)

    #COLLECT RESULTS
    results_df = pd.DataFrame(
    [(q, a, t) for q, (a, t) in results.items()],
    columns=['problem', 'y_pred', 'time_seconds'])
    results_df = results_df.merge(df[['problem', 'parsed_answer']], on = 'problem', how = 'left')
    results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
    results_df['correct'] = results_df.apply(check_correct, axis=1)
    results_df.to_csv(f'/mnt/d/math_ttc_search/results/mcts_MATH_1.5B_{num_iters}_iters_final.csv', index=False)
  
  