# math_ttc_search

Test-time compute scaling on MATH-500, using Qwen2.5-1.5B-Instruct with vLLM and Qwen2.5-Math-PRM-7B as a process reward model. This project tests whether allocating more compute at inference time (sampling, search) can substitute for model scale, and where that substitution breaks down.

The primary motivation was an exploration of the ideas raised in the HuggingFace blog post [Scaling Test-Time Compute with Open Models](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) (the Search-and-Learn library), and more deeply in the DeepMind paper [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314). My aim was a partial replication, particularly in the beam search sections, where rather than scoring through a Process Reward Model (PRM) throughout, I primarily score using standard beam search (SBS) selection criteria, logprob-based rather than value-guided. This was partly a deliberate choice: running a 7B PRM alongside the generation model roughly doubles VRAM requirements, and much of this project's practical constraints came from exactly that tradeoff. My analysis also records wall-clock time per problem across every method, building out an accuracy-vs-time Pareto frontier (see Key Findings) as a dimension neither source work isolates on its own. Beyond SBS, my methods include probabilistic and diverse beam search variants, and a Best-of-N baseline. vLLM was used, mainly for its parallelization capabilities being critical in speeding up inference time in beam search and Best-of-N runs.

The LLM choices are focused on Alibaba's Qwen Instruct series, with the 1.5B variant as the primary search-augmented model. As a smaller model, this is aligned with the broader theme of the reference work: testing whether relatively inexpensive adjustments in inference can partially substitute for raw parameter count.

## What's here

Four inference-time strategies evaluated on the same 458-problem MATH-500 subset:

- **Baseline**: single-shot greedy decoding (7B model, matched-budget comparison at 400 tokens, plus a separate 800-token run to isolate the effect of generation budget)
- **Best-of-N**: N independent samples (1.5B model), majority vote over parsed answers, N ∈ {1, 2, 4, 8, 16}
- **Beam search**: standard, diverse, and probabilistic selection variants (1.5B model), same N sweep
- **PRM-guided beam search**: value-guided selection using Qwen2.5-Math-PRM-7B as the scoring function (partial results, see Limitations)

## Key findings

### Accuracy vs N
<img width="1350" height="900" alt="image-3" src="https://github.com/user-attachments/assets/3ea2d045-96ab-4dc7-a2ae-e41247bff29f" />

**Search on a smaller model closes the gap to a larger single-shot model.** At matched 400-token generation budget, the 7B baseline scores 46.7% single-shot. Best-of-N on the 1.5B model starts well below that at N=1 (33.2%) but climbs monotonically to 49.6% by N=16, surpassing the 7B baseline outright with a model roughly 4.6x smaller. This is the core empirical claim in Snell et al.: that test-time compute, spent well, can be more effective than spending the equivalent compute on parameters. The Search-and-Learn blog demonstrates the same shape of result with Llama 3.2 1B/3B closing the gap to Llama 3.1 8B; this project reproduces that pattern on a different model family (Qwen2.5) and a different scoring signal (majority vote over raw samples rather than a PRM-weighted vote), which is itself informative, as 1. the effect isn't specific to one model family or one verification method and 2. the actual test-time computation is a cheaper operation than running a larger PRM based method. This points to a potentially cheaper alternative to LLM-based reward methods like a PRM: ranking beams by cumulative logprobs instead still recovers meaningful gains, while staying entirely within the 1.5B model's own compute budget, with no second model required thereby saving on the total FLOPs required for a single problem along with the natural VRAM savings.

### Wall Clock vs N
<img width="1350" height="900" alt="image-4" src="https://github.com/user-attachments/assets/ef96868f-56a0-4492-bdc5-7c8c58ab7e92" />

**Wall-clock cost scales very differently by method** Best-of-N's cost barely grows until N=16 (1.16s → 3.81s from N=1 to N=8), since independent samples batch essentially for free under vLLM. Every beam variant instead grows close to linearly with N from the very start, standard and probabilistic both roughly triple in cost from N=1 to N=16, because each additional beam means another sequential generation round, not another item in a batch. Diverse beam is the steepest of the three (12.99s → 75.74s, N=1 to N=16), which tracks directly to its implementation: the diversity penalty compares each candidate against every existing beam's `text_so_far` via `SequenceMatcher`, an O(beam_count²) cost per round that best-of-N and the other beam variants don't require.

**Diverse beam search peaks at N=8, higher than every other beam variant ever reaches, then declines at N=16 much like standard beam does.** At matched budget, diverse beam improves close to monotonically through N=8 (42.4% → 41.0% → 45.4% → 46.5%, closing in on but not quite reaching the 7B baseline), but drops to 44.98% at N=16. This peak-then-decline shape is shown with SBS as well (43.45% at N=8, down to 42.58% at N=16), just at a meaningfully higher peak and a shallower decline. Probabilistic selection shows no consistent trend across the same sweep. Evidently, diversity doesn't prevent the eventual decline unguided beam search shows at high N, it delays it and raises the ceiling before it hits. This finding is consistent with the Search-and-Learn team's own reasoning for building DVTS (an extension of beam search I partially mirror with my diverse variant, which splits the initial beams into independent subtrees, each expanded greedily using a PRM): diversity alone recovers some of what a real verifier would catch, but not all of it, more beams without a strong value signal still eventually surfaces more bad branches than good ones.

**Wall-clock is a dimension that particularly favors best-of-N at low latency.** This is my specific extension beyond both source works: recording per-problem generation time for every method and N, then plotting accuracy against it directly.

### Wall Clock vs Accuracy (Pareto Frontier)
<img width="1350" height="975" alt="image-2" src="https://github.com/user-attachments/assets/aba41de5-1b04-4494-8fc0-20ac359b44fa" />

At this budget, best-of-N wins the frontier decisively at both ends but not cleanly in the middle. At N=16 (49.6% accuracy, 10.2s/problem) it beats every single beam search configuration on both axes simultaneously, more accurate than even diverse beam's own peak (46.5% at N=8) and still faster than the cheapest beam config tested (11.8s/problem). At N=8, best-of-N (44.3%, 3.8s/problem) remains dramatically faster than any beam variant, an order of magnitude or more, but is no longer the most accurate option at that point, diverse beam's N=8 result edges past it by a couple of points. The frontier isn't uniformly dominated by one method at every budget, best-of-N wins on cost everywhere and wins outright once N is large enough, while diverse beam holds a narrow, temporary accuracy edge in the middle of the range before its own decline at N=16. That's consistent with the scoring-signal explanation above: paying for extra sequential generation rounds only pays off if what's guiding the selection between rounds is worth the cost, and here that's a close call in the middle of the budget range and a clear loss at both ends. 

## Results

| Method | N | Accuracy | Non-completion |
|---|---|---|---|
| Baseline (7B, 400 tok) | 1 | 46.7% | 44.3% |
| Baseline (7B, 800 tok) | 1 | 69.2% | 13.8% |
| Best-of-N (1.5B) | 1 | 33.2% | 10.3% |
| Best-of-N (1.5B) | 16 | 49.6% | 0.0% |
| Standard beam | 1 | 38.9% | 18.1% |
| Standard beam | 16 | 42.6% | 27.1% |
| Diverse beam | 1 | 42.4% | 19.4% |
| Diverse beam | 16 | 45.0% | 13.3% |
| Probabilistic beam | 1 | 42.4% | 11.6% |
| Probabilistic beam | 16 | 42.1% | 6.8% |

## Future Extensions & Limitations

**Token budget is a first-order factor in how much a model can demonstrate** Bumping the baseline's generation cap from 400 to 800 tokens raised overall accuracy from 46.7% to 69.2%, while accuracy conditional on completing a problem barely moved (83.9% vs 80.3%). The model's per-problem correctness was fairly stable either way, what changed was how often it had room to finish. This scaled with problem difficulty (roughly 10% non-completion at Level 1 up to 70% at Level 5 at 400 tokens) and reproduced across every method tested. Snell et al. frame compute-optimal test-time scaling as a question of how to *allocate* a fixed budget, sampling width, search depth, revision steps. This result adds a variable underneath all of those: the length of a single trajectory is itself a compute allocation decision, and here it dominated everything downstream of it.

PRM-guided beam search and MCTS were both implemented but not run to completion. This is mainly due to compute & budget constraints, however the code is presented and capable for running at a later time.

## Repo structure

```
src/
  search/
    baseline_7B.py
    best_of_N.py
    beam_search.py       # standard, diverse, probabilistic
    value_guided.py       # PRM-guided
    mcts.py                # implemented, not completed
    results/
  eval/
    load_and_parse.py
  serve_vllm/
    model.py
```

## Setup

```bash
pip install vllm pandas datasets math_verify scipy transformers accelerate
```

The code has been optimized to work on an RTX 2060. Triton vLLM kernels are used for this reason (Turing architecture, sm75, lacks FlashAttention support).

## Author

Mukund Hari
mukund.hari0@gmail.com
