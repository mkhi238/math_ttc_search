#IMPORTS
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from eval.load_and_parse import make_math_parser, load_dataset
from serve_vllm.model import load_vllm
from vllm import SamplingParams
import pandas as pd
from math_verify import parse, verify
import time
import random

#PROMPT
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

#PARAMETERS
N = [1, 2, 4, 8, 16]
TEMPRATURE = 0.75
MAX_TOKENS = 400
CHECKPOINT = 200
SIZE = 1.5

def create_sampling_parameters(n=N, temperature=TEMPRATURE, max_tokens=MAX_TOKENS):
    sampling_parameters = SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens)
    return sampling_parameters
  
def format_prompt(question, pool = FEW_SHOT_EXAMPLES, k = 5):
  items = random.sample(pool, k)
  example_txt = "\n\n".join(items)

  return (INTRO + "\n\n" + 
    example_txt + 
    f"\nProblem: {question}\n"
    "Step 1:"
  )

def generate_responses(df, llm, samples):
  response_dict = {}
  timing_dict = {}
  for idx, question in enumerate(df['problem']):
    start_time = time.time()
    prompts = format_prompt(question)
    outputs = llm.generate(prompts, samples)
    timing_dict[question] = time.time() - start_time
    response_dict[question] = outputs

    if (idx + 1) % CHECKPOINT == 0:
      print(f"Checkpoint: {idx+1} completed")
  return response_dict, timing_dict

def check_correct(row):
  if row['y_pred'] is None:
    return 0
  try:
    return int(verify(row['y_true'], row['y_pred']))
  except Exception:
    return 0

def majority_vote(question, candidate): 
  completions = candidate[question][0].outputs # CompletionOutput(idx, text, token_ids, logprobs,...)
  parsed_list = []
  for o in completions:
    parsed = parse(o.text)
    if not parsed:
      continue
    parsed_list.append(parsed)
  if not parsed_list:
    return None
  seen = {}
  scores = [1]*len(parsed_list)
  for p in parsed_list:
    for s in range(len(seen)):
      if verify(p, seen[s]):
        scores[s] += 1
        break
    else:
      seen[len(seen)] = p
      
  best_idx = scores.index(max(scores))
  return seen[best_idx]


if __name__ == "__main__":
  
  #LOAD VLLM
  llm = load_vllm(f"Qwen/Qwen2.5-{SIZE}B-Instruct", 'float16', 0.8)

  #LOAD DATA
  ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
  df = pd.DataFrame(ds)
  df = make_math_parser(df, 'answer', 'parsed_answer')
  undefined_count = df['parsed_answer'].apply(lambda p: len(p) == 0).sum()
  print(f"{undefined_count} / {len(df)} rows failed to parse at all")
  
  #GENERATE SAMPLES  
  for idx in N:
    samples = create_sampling_parameters(n = idx, temperature = TEMPRATURE, max_tokens = MAX_TOKENS)
    candidates, timing_dict = generate_responses(df, llm, samples)
    #EXTRACT
    results = {}
    for q in df['problem']:
      results[q] = majority_vote(q, candidates)

    #COLLECT RESULTS
    results_df = pd.DataFrame(list(results.items()), columns=['problem', 'y_pred'])
    results_df['time_seconds'] = results_df['problem'].map(timing_dict)
    results_df = results_df.merge(df[['problem', 'parsed_answer', 'level', 'subject']], on = 'problem', how = 'left')
    results_df = results_df.rename(columns={'parsed_answer': 'y_true'})
    results_df['correct'] = results_df.apply(check_correct, axis=1)
    results_df.to_csv(f'results/best_of_n_MATH_1.5B_{idx}_iterations.csv', index=False)

  
  