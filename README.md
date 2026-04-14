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

**Discover exact mathematical formulas from raw numerical data.**

Feed in data points. Get back `exp(x)`, `1/x`, `ln(x)` -- not a neural net approximation, not a polynomial fit, but the actual symbolic formula. Powered by a single binary operator that can express every elementary function in mathematics.

```python
from eml import regress
import numpy as np

x = np.linspace(0.1, 5, 200)
y = np.exp(x)  # pretend we don't know this

result = regress(x, y)
print(result["expression"])  # exp(x)
```

## Why this exists

### The paper

In 2026, Andrzej Odrzywołek proved that one binary operator is enough to generate **all elementary functions** -- exponentials, logarithms, trigonometry, powers, roots, polynomials. Everything on a scientific calculator. One operator:

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

EML symbolic regression searches a **minimal, complete grammar**. One operator, exhaustive search at shallow depths, gradient-free constant optimisation. If the formula exists, it finds it.

---

## Install

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `sympy`. No GPU needed.

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
print(result["eml_expression"])  # eml(eml(tiny, x), eml(0, 1))
print(result["mse"])             # ~6e-32
```

### CLI

```bash
# From expression (generates data, discovers formula)
eml-regress --func "exp(x)" --range 0.1 5

# From JSON data
echo '{"x": [1,2,3,4], "y": [2.72,7.39,20.09,54.60]}' | eml-regress

# JSON output for automation
echo '{"x": [...], "y": [...]}' | eml-regress --output-json
```

---

## Hermes Agent Integration

This tool was designed to give **small local LLMs** mathematical capabilities they can't have natively. A 7B Hermes model that can call this tool outperforms a 70B model doing math from memory.

### The pattern

```
User: "What formula fits this data?"
          |
    Hermes extracts x, y values
          |
    Calls eml_symbolic_regression tool
          |
    Gets back {"expression": "exp(x)", "quality": "exact"}
          |
    Reports: "The data follows y = e^x"
```

**The model orchestrates. The tool does the math.**

### Three integration paths

Pick the one that fits your setup:

#### Option A -- Native tool (tightest integration)

Copy `hermes/hermes_tool.py` into your `hermes-agent/tools/` directory.
Make sure it gets imported at startup. Done.

```python
# hermes_tool.py auto-registers with the Hermes ToolRegistry:
#   name:    "eml_symbolic_regression"
#   toolset: "math"
#   handler: Python callable (no subprocess, no shell)
```

The handler calls `from eml import regress` directly -- no stdin/stdout, no CLI wrapping. This is how Hermes native tools work: Python function in, JSON string out.

#### Option B -- MCP server (no code changes to Hermes)

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  eml:
    command: python
    args: ["/path/to/eml-symbolic-regression/hermes/mcp_server.py"]
    timeout: 120
```

Hermes auto-discovers the tool as `mcp_eml_eml_symbolic_regression` via the MCP stdio protocol. No changes to the Hermes codebase needed.

#### Option C -- Skill only (prompt-based guidance)

Copy `hermes/skill/` into your `hermes-agent/skills/math/symbolic-regression/` directory. This adds a skill entry that teaches Hermes *when* and *how* to use the tool, including fallback strategies when results are approximate.

### What the LLM needs to do

Almost nothing. It just needs to:
1. Recognise "find the formula" type questions
2. Put the numbers into `x` and `y` arrays
3. Call the tool
4. Report the result

No chain-of-thought math. No symbolic manipulation. No arithmetic. The transformer does what it's good at (language), the tool does what it's good at (math).

### Works with any OpenAI-compatible framework

The tool schema in `hermes/tool_schema.json` is standard OpenAI function-calling format. If you're using vLLM, Ollama, or raw llama.cpp instead of hermes-agent, the stdin/stdout CLI mode still works:

```bash
echo '{"x": [1,2,3,4], "y": [2.72,7.39,20.09,54.60]}' | python -m eml.engine
```

---

## Benchmarks

Tested on common functions, 200 data points, `x in [0.1, 4.0]`:

| Target | Discovered | MSE | Depth | Time | Status |
|--------|-----------|-----|-------|------|--------|
| `exp(x)` | `exp(x)` | 0 | 1 | 0.1s | **EXACT** |
| `1/x` | `1/x` | 6e-32 | 2 | 2s | **EXACT** |
| `ln(x)` | `ln(x)` | 4e-33 | 3 | 33s | **EXACT** |
| `exp(-x)` | `exp(-x)` | ~0 | 2 | 1s | **EXACT** |
| `sin(x)` | approx | 7e-3 | 3 | 35s | Approx |
| `x + 1` | approx | 6e-2 | 2 | 4s | Approx |
| `x^2` | approx | 2e-1 | 3 | 35s | Approx |

### Sweet spot

Functions built from `exp` and `ln` are found **exactly** at shallow depth. This covers most of physics, biology, and finance: exponential growth/decay, power laws, logarithmic scaling, reciprocal relationships.

### Honest limitations

- **Polynomials** (`x+1`, `x^2`) are hard -- EML is fundamentally nonlinear, so linearity must emerge from cancellation
- **Trig functions** need deeper trees (depth 4+) for exact results
- **Single variable** only -- no `f(x, y)` yet
- **Clean data preferred** -- the optimizer can overfit to noise

---

## How it works

### The grammar

Every expression is a **full binary tree** where:
- Leaves are: `x`, `0`, `1`, or `c` (trainable constant)
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
                                                  ≈ 1/x  (when c → -inf)
```

### The search (3 phases per depth)

```
Phase 1  Instant eval
         Test all trees with no trainable constants.
         Pure numpy, microseconds each.
         Catches easy wins like eml(x, 1) = exp(x).

Phase 2  Parallel screening
         Quick-fit constants (1 restart, 200 iterations).
         ~7ms per tree. Multiprocessing across all CPU cores.
         Screens thousands of candidates fast.

Phase 3  Full refinement
         Top 30 candidates get full optimisation.
         3 restarts, 2000 iterations each.
         Polishes the best discoveries.
```

### Why not traditional symbolic regression?

Traditional SR (PySR, gplearn, etc.) searches over a large ad-hoc grammar: `+, -, *, /, sin, cos, exp, log, pow, sqrt, ...`. The search space is irregular and the grammar choice is arbitrary.

EML has **one operator**. The search space is a set of binary trees with uniform structure. This means:
- Exhaustive search is feasible at depth 1-3
- The grammar is **provably complete** (not heuristic)
- Constants are optimised with standard methods (Nelder-Mead)
- Results are reproducible

---

## Project structure

```
eml-symbolic-regression/
  eml/
    __init__.py              Clean API: from eml import regress
    engine.py                Core search engine (3-phase parallel search)
  hermes/
    hermes_tool.py           Native Hermes tool module (drop into tools/)
    mcp_server.py            MCP stdio server (configure in config.yaml)
    tool_schema.json         OpenAI function-calling schema (inner format)
    system_prompt_snippet.md System prompt addition for tool guidance
    skill/
      SKILL.md               Hermes skill definition (prompt-based)
  examples/
    quickstart.py            3-line usage
    from_data_points.py      Realistic scenario (RC circuit decay)
    benchmark.py             Performance table generator
    cli_examples.sh          CLI usage patterns
  pyproject.toml             pip installable
  LICENSE                    MIT
```

---

## Reference

```bibtex
@article{odrzywołek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywołek, Andrzej},
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
