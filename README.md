<p align="center">
  <br>
  <code>eml(a, b) = exp(a) - ln(b)</code>
  <br><br>
  <strong>One operator. Every function.</strong>
  <br>
  <em>The NAND gate of continuous mathematics.</em>
  <br><br>
  <a href="#install">Install</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#hermes-integration">Hermes / LLM</a> &bull;
  <a href="#how-it-works">How it works</a> &bull;
  <a href="#benchmarks">Benchmarks</a>
</p>

---

# EML Symbolic Regression

Discover exact mathematical formulas from raw numerical data.

Feed in data points. Get back `exp(x)`, `1/x`, `ln(x)`, `x0 + x1`, or `exp(x0) - log(x1)` -- not a neural net approximation, not a polynomial fit, but the actual symbolic formula. The engine uses a hybrid exact-form pre-pass before falling back to EML tree search, so simple closed forms return immediately while harder functions still use the uniform EML grammar.

```python
from eml import regress
import numpy as np

x = np.linspace(0.1, 5, 200)
y = np.exp(x)  # pretend we do not know this

result = regress(x, y)
print(result["expression"])  # exp(x)

# Reproducible search when you want stable comparisons
result = regress(x, y, seed=123)

# Or let the analyzer pick a search budget
result = regress(x, y, mode="auto")
```

## Why this exists

### The paper

In 2026, Andrzej Odrzywolek proved that one binary operator is enough to generate all elementary functions -- exponentials, logarithms, trigonometry, powers, roots, polynomials. Everything on a scientific calculator. One operator:

```
eml(a, b) = exp(a) - ln(b)
```

Combined with `{x, 0, 1}` and the grammar `S -> x | 0 | 1 | c | eml(S, S)`, every expression becomes a binary tree of identical nodes. This is the continuous-math equivalent of the NAND gate in Boolean logic.

> Paper: [arXiv:2603.21852](https://arxiv.org/abs/2603.21852)

### The problem it solves

You have data. You need the formula. Traditional options:

| Method | Issue |
|--------|-------|
| Polynomial regression | Misses exp, log, trig |
| Neural networks | Black box, no symbolic output |
| GP symbolic regression (PySR etc.) | Large grammar, slow, non-deterministic |
| Ask an LLM | Hallucinated arithmetic, wrong answers |

EML symbolic regression searches a minimal, complete grammar. One operator, exhaustive or stratified search at shallow depths, gradient-free constant optimisation, and a cheap closed-form pre-pass for standard families.

---

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `sympy`. No GPU needed.

---

## Current Status

- Hybrid exact-form pre-pass for common 1D and multivariable families before EML tree search
- 1D vector input plus matrix input with up to 3 features and optional custom `feature_names`
- Deterministic search with `seed=` for repeatable comparisons
- Native Hermes tool integration plus skill-only guidance
- Dedicated multivariable benchmark script at `examples/benchmark_multivariable.py`

---

## Quickstart

### Python

```python
from eml import regress
import numpy as np

# Your data (or measurements, or simulation output)
x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
y = np.array([2.0, 1.0, 0.667, 0.5, 0.4, 0.333, 0.286, 0.25])

result = regress(x, y, max_depth=2)

print(result["expression"])      # 1/x
print(result["strategy"])        # prepass
print(result["depth"])           # 0  (no EML tree was needed)
print(result["eml_expression"])  # None for prepass hits
print(result["mse"])             # ~6e-32
```

Two-variable matrix input works too:

```python
from eml import regress
import numpy as np

x0, x1 = np.meshgrid(np.linspace(0.1, 2.0, 8), np.linspace(0.5, 2.5, 6))
X = np.column_stack([x0.ravel(), x1.ravel()])
y = X[:, 0] + X[:, 1]

result = regress(X, y)

print(result["expression"])     # x0 + x1
print(result["feature_names"])  # ['x0', 'x1']
print(result["used_features"])  # ['x0', 'x1']
```

Custom feature names are supported:

```python
result = regress(X, y, feature_names=["length", "time"])
print(result["expression"])  # length + time
```

Result fields:

- `expression`: simplified symbolic expression to show the model or user
- `strategy`: `"prepass"` for closed-form hits, `"eml_tree"` for actual EML-tree discoveries
- `depth`: EML tree depth only; `0` means the pre-pass solved it before tree search
- `eml_expression`: raw EML tree string, or `None` when the pre-pass found the answer
- `mse`, `constants`, `leaf_types`: fit quality and low-level search details
- `feature_names`, `used_features`: resolved input names and which features appear in the final formula
- `analysis`: routing hints like likely family, recommended search mode, and feature importance
- `verification`, `confidence`, `failure_modes`: trust checks for LLM/tool use
- `candidates`: ranked alternative formulas with short rationales
- `guidance`: compact LLM-facing summary with caveats and follow-up experiment suggestions

For cheap routing before any search, call:

```python
from eml import analyze

analysis = analyze(x, y)
print(analysis["recommended_mode"])         # instant / balanced / deep
print(analysis["likely_families"][0])       # cheap heuristic classifier
print(analysis["should_attempt_regression"])
```

Search presets:

- `instant`: shallow, fast checks for cheap wins
- `balanced`: current default tradeoff for normal use
- `deep`: larger search budget for harder symbolic fits
- `research`: heaviest built-in budget
- `auto`: follow `analyze(...)[\"recommended_mode\"]`

### CLI

```bash
# From expression (generates data, discovers formula)
eml-regress --func "exp(x)" --range 0.1 5

# Deterministic search
eml-regress --func "sin(x)" --range 0.1 5 --seed 123

# Auto-budgeted search
eml-regress --func "exp(x)" --range 0.1 5 --mode auto

# From JSON data
echo '{"x": [1,2,3,4], "y": [2.72,7.39,20.09,54.60]}' | eml-regress

# From 2-feature JSON data
echo '{"x": [[0.1,0.5],[0.2,0.5],[0.1,0.7],[0.2,0.7]], "y": [0.6,0.7,0.8,0.9]}' | eml-regress

# From 3-feature JSON data
echo '{"x": [[1,2,1],[2,2,1],[1,3,2],[2,3,2]], "y": [2,4,0.75,1.5], "feature_names": ["m1","m2","r"]}' | eml-regress

# With custom feature names
echo '{"x": [[0.1,0.5],[0.2,0.5],[0.1,0.7],[0.2,0.7]], "y": [0.6,0.7,0.8,0.9], "feature_names": ["a","b"]}' | eml-regress

# JSON output for automation
echo '{"x": [...], "y": [...]}' | eml-regress --output-json
```

JSON output includes the full structured result object, including `analysis`, `verification`, `confidence`, `failure_modes`, `candidates`, `guidance`, `requested_mode`, `effective_mode`, and `search_budget`.

---

## Hermes Integration

This tool was designed to give small local LLMs mathematical capabilities they cannot have natively. A 7B Hermes model that can call this tool can outperform a much larger model trying to do the math from memory.

### The pattern

```
User: "What formula fits this data?"
          |
    Hermes extracts x, y values
          |
    Calls eml_symbolic_regression tool
          |
    Gets back {"expression": "exp(x)", "quality": "exact", "strategy": "prepass", "confidence": {"label": "high"}}
          |
    Reports: "The data follows y = e^x"
```

The model orchestrates. The tool does the math.

### Two integration paths

#### Option A -- Native tool

Three steps:

1. Copy `hermes/hermes_tool.py` into `hermes-agent/tools/eml_tool.py`
2. Register the import in `tools/model_tools.py` by adding `"tools.eml_tool"` to `_discover_tools()`
3. Add `"eml_symbolic_regression"` to the relevant Hermes toolset in `tools/toolsets.py`

The handler calls `from eml import regress` directly and uses the Hermes registry helpers.

Native Hermes responses include:

- `expression`: best symbolic form to show the user
- `quality`: `exact`, `approximate`, or `rough`
- `analysis`: routing hints so a small model can decide whether the call was worthwhile
- `confidence`: compact trust score plus reasons
- `verification`: post-search checks and failure signals
- `failure_modes`: warnings like piecewise / periodic / unstable constants / outlier dominance
- `candidates`: ranked alternative formulas for model-side comparison
- `guidance`: why the winner won, when not to trust it, and what follow-up data to collect
- `strategy`: `prepass` or `eml_tree`
- `depth`: EML depth only; `0` means the hybrid pre-pass solved it
- `mse`, `constants`, `eml_expression`, `feature_names`, `used_features`

Python and CLI calls also expose `seed` for deterministic search plus `mode` for budgeted search. The Hermes native wrapper exposes `x`, `y`, optional `feature_names`, `max_depth`, and `mode`, where `x` can be a 1D array or a 2D matrix.

#### Option B -- Skill only

Copy the `hermes/skill/` directory into `hermes-agent/skills/math/eml-symbolic-regression/`. The SKILL file teaches Hermes when and how to use the tool, including fallback guidance when results are approximate.

### What the LLM needs to do

Almost nothing. It just needs to:

1. Recognise "find the formula" type questions
2. Put the numbers into `x` and `y` arrays
3. Call the tool
4. Report the result

No chain-of-thought math. No symbolic manipulation. No arithmetic. The transformer does what it is good at (language), the tool does what it is good at (math).

### Works with any OpenAI-compatible framework

The schema in `hermes/tool_schema.json` uses standard OpenAI function-calling format. If you are using vLLM, Ollama, or raw llama.cpp instead of hermes-agent, stdin/stdout CLI mode still works:

```bash
echo '{"x": [1,2,3,4], "y": [2.72,7.39,20.09,54.60]}' | python -m eml.engine
```

On Windows, multiprocessing works best from a normal script or agent process. For ad-hoc REPL or `python -` experiments, prefer the CLI path above or force single-process Python calls with `workers=1`.

---

## Benchmarks

Benchmark scripts:

- `python examples/benchmark.py`
- `python examples/benchmark_multivariable.py`

Single-variable benchmark on common functions, 200 data points, `x in [0.1, 4.0]`:

| Target | Discovered | MSE | Depth | Time | Status |
|--------|-----------|-----|-------|------|--------|
| `exp(x)` | `exp(x)` | ~0 | 0 | 0.1s | EXACT |
| `1/x` | `1/x` | ~0 | 0 | 0.0s | EXACT |
| `ln(x)` | `log(x)` | ~0 | 0 | 0.1s | EXACT |
| `exp(-x)` | `exp(-x)` | ~0 | 0 | 0.1s | EXACT |
| `x + 1` | `x + 1` | ~0 | 0 | 0.0s | EXACT |
| `x^2` | `x**2` | ~0 | 0 | 0.0s | EXACT |
| `sqrt(x)` | `sqrt(x)` | ~0 | 0 | 0.1s | EXACT |
| `sin(x)` | approx | `1.1e-2` | 3 | 32.0s | Approx |

`Depth = 0` means the hybrid pre-pass found an exact closed form before any EML tree search ran. It does not mean there is a zero-leaf EML tree.

Two-variable benchmark on a small `(x0, x1)` grid:

| Target | Discovered | MSE | Strategy | Depth | Time |
|--------|-----------|-----|----------|-------|------|
| `x0 + x1` | `x0 + x1` | ~0 | prepass | 0 | 0.1s |
| `x0 * x1` | `x0*x1` | ~0 | prepass | 0 | 0.1s |
| `x0 / x1` | `x0/x1` | ~0 | prepass | 0 | 0.1s |
| `exp(x0 + x1)` | `exp(x0 + x1)` | ~0 | prepass | 0 | 0.1s |
| `x0^2 + x1` | `x0**2 + x1` | ~0 | prepass | 0 | 0.1s |
| `eml(x0, x1)` | `exp(x0) - log(x1)` | 0 | eml_tree | 1 | 0.2s |

### Sweet spot

Common exponentials, logarithms, power laws, low-degree polynomials, and several basic multivariable forms are found exactly and usually return before the EML tree search even starts. This covers exponential growth/decay, reciprocal relationships, roots, affine multivariable forms, simple products and ratios, and standard algebraic forms.

### Honest limitations

- Oscillatory / trig functions (`sin`, `cos`) still need deeper trees (depth 4+) for exact recovery
- Up to 3 variables in the current tree search -- deeper multivariable scaling still needs guided search
- Clean data preferred -- the optimiser can overfit noise
- Sparse or narrow-range data can be misleading -- a low-error polynomial surrogate may beat the intended closed form on small samples
- Hermes `quality` is heuristic -- it is based on MSE thresholds, not a proof that the recovered expression is the unique underlying law

---

## How it works

### The grammar

Every expression is a full binary tree where:

- Leaves are feature names (`x`, `x0`, `x1`, ...), `0`, `1`, or `c` (trainable constant)
- Internal nodes are all `eml(left, right)`

```
        eml              depth 1: 2 leaves
       /   \
      x     1            eml(x, 1) = exp(x) - ln(1) = exp(x)
```

```
        eml              depth 2: 4 leaves
       /   \
     eml   eml
    / \    / \
   c   x  0   1         eml(eml(c,x), eml(0,1)) = exp(exp(c)-ln(x)) - ln(1)
                                                  ~= 1/x  (when c -> -inf)
```

### The search

```
Phase 0  Hybrid exact-form pre-pass
         Try cheap closed-form fits before touching the EML tree search.
         1D: polynomial (deg 1-4), power law, exponential, logarithmic.
         2D/3D: affine, interaction, quadratic, power-law, exponential, and log-linear fits.
         Solves x + 1, x^2, sqrt(x), 1/x, exp(x), ln(x), x0 + x1, x0*x1, and m1*m2/r^2 in milliseconds.
```

```
Phase 1  Instant eval
         Test all trees with no trainable constants.
         Uses subtree caching and duplicate filtering for constant-free trees.
         Pure numpy, microseconds each.
         Catches easy wins like eml(x, 1) = exp(x).

Phase 2  Parallel screening
         Quick-fit constants (1 restart, adaptive iteration budget).
         Prunes redundant constant-only substructures.
         Uses stratified sampling plus multiprocessing for large searches.
         Screens thousands of candidates fast.

Phase 3  Full refinement
         Top 30 candidates get full optimisation.
         Broader initialisation scales.
         Polishes the best discoveries.
```

### Why not traditional symbolic regression?

Traditional symbolic regression systems (PySR, gplearn, etc.) search over a large ad-hoc grammar: `+, -, *, /, sin, cos, exp, log, pow, sqrt, ...`. The search space is irregular and the grammar choice is arbitrary.

EML has one operator. The search space is a set of binary trees with uniform structure. This means:

- Exhaustive search is feasible at depth 1-3
- The grammar is provably complete, not heuristic
- Constants are optimised with standard methods such as Nelder-Mead
- Results can be made reproducible with `seed=` or `--seed`

---

## Project structure

```
eml-symbolic-regression/
  eml/
    __init__.py                 Clean API: from eml import regress
    engine.py                   Hybrid pre-pass + EML search engine
  hermes/
    hermes_tool.py              Native Hermes tool module
    tool_schema.json            OpenAI function-calling schema
    system_prompt_snippet.md    System prompt guidance
    skill/
      SKILL.md                  Hermes skill definition
  examples/
    quickstart.py               3-line usage
    from_data_points.py         Noisy measurement example
    benchmark.py                Single-variable benchmark generator
    benchmark_multivariable.py  Multivariable benchmark generator
    cli_examples.sh             CLI usage patterns
  tests/
    test_engine.py              Regression + symbolic/numeric consistency checks
  pyproject.toml                pip-installable project metadata
  LICENSE                       MIT
```

---

## Reference

```bibtex
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywolek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}
```

---

<p align="center">
  <strong>Give small models big math.</strong>
  <br>
  <code>pip install -e .</code>
</p>
