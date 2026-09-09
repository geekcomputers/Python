# Bug Report: `estimate_complexity()` and `reduceMax` — confirmed issues

> Prepared as a constructive code review. All findings are reproducible with minimal
> snippets included. No changes to logic are proposed — only the bugs are documented.

---

## 1. `ML/src/python/neuralforge/nas/search_space.py` — `estimate_complexity`

### 1a. `UnboundLocalError` when genome starts with an `identity` gene

**Location:** `estimate_complexity()`, the `for gene in architecture.genome` loop
(lines ~159-179 in the current `master`).

The variables `params` and `flops` are only assigned inside the `if/elif` branches for
`conv3x3/conv5x3/conv7x7/depthwise/bottleneck`. If the **first** gene in the genome is
`identity` (or `pooling`), none of those branches execute, so the subsequent line:

```python
total_params += params   # ← NameError / UnboundLocalError
```

references a name that has never been bound.

**Reproduction (confirmed by running `search_space.py` directly with a stubbed `torch`):**

```python
ss = SearchSpace({})
genome = [{"type": "identity", "channels": 64,
           "activation": "relu", "use_bn": False, "dropout": 0.0}]
ss.estimate_complexity(Architecture(genome))
# UnboundLocalError: cannot access local variable 'params'
#                     where it is not associated with a value
```

**Suggested fix:** initialize `params = 0` and `flops = 0` at the top of the loop body
(before the `if` chain), so that every code path has a defined value.

---

### 1b. Stale `params`/`flops` carried across genes → double-count

Even when the first gene is a conv (so no `UnboundLocalError`), the variables are never
reset between loop iterations. An `identity` gene after a `conv3x3` therefore re-adds the
**previous conv's** `params` and `flops`.

**Reproduction (confirmed by running):**

```python
genome = [
    {"type": "conv3x3",  "channels": 64,  "activation": "relu",
     "use_bn": False, "dropout": 0.0},
    {"type": "identity", "channels": 64,  "activation": "relu",
     "use_bn": False, "dropout": 0.0},
]
r = ss.estimate_complexity(Architecture(genome))
# r == {'params': 3456, 'flops': 173408256}
# correct params for a single 3x64x3x3 conv = 3*64*3*3 = 1728
# → over-counted by exactly 2×
```

**Suggested fix:** same as 1a — set `params = 0; flops = 0` at the top of each loop
iteration.

---

### 1c. `identity` with a channel change silently drops the 1×1 conv parameters

In `build_model()`, an `identity` gene whose input and output channel counts differ is
implemented as a `nn.Conv2d(current, out, 1)` (a 1×1 conv with `in*out` parameters).
`estimate_complexity()` has **no branch** for this case and reports the parameters as
whatever stale value was left over (see 1b) or 0 (see 1a).

**Reproduction (confirmed by running):**

```python
genome = [
    {"type": "conv3x3",  "channels": 64,  "activation": "relu",
     "use_bn": False, "dropout": 0.0},
    {"type": "identity", "channels": 128, "activation": "relu",
     "use_bn": False, "dropout": 0.0},
]
r = ss.estimate_complexity(Architecture(genome))
# The identity step inserts a Conv2d(64, 128, 1)  →  64*128 = 8192 params
# correct total = 1728 + 8192 = 9920
# reported      = 3456   (stale conv value re-added, 1×1 conv params missing)
```

**Suggested fix:** add an `elif gene['type'] == 'identity':` branch that computes
`params = current_channels * out_channels` when the two differ, else 0.

---

## 2. `ML/src/cuda/kernels.cu` — `reduceMax`

### 2a. `atomicMax` on `__float_as_int` is only correct for non-negative floats

**Location:** `reduceMax`, line ~170.

```cuda
if (tid == 0) {
    atomicMax((int*)output, __float_as_int(sdata[0]));
}
```

`__float_as_int` reinterprets the bit pattern of a `float` as an `int`. For **non-negative**
floats the integer ordering matches the float ordering (both are monotonic from 0 upward),
so `atomicMax` on the int representation works.

For **negative** floats the sign bit is set, so the integer value is negative and the
ordering is **reversed** relative to float order. Example:

| float  | `__float_as_int` (hex) | as signed int |
|--------|------------------------|---------------|
| −1.0   | `0xBFC00000`           | −1088259840   |
| −0.5   | `0xBF000000`           | −1072693248   |

`-1.0 < -0.5` in float, but `-1088259840 < -1072693248` is also true here — so for two
negatives it happens to work. However, mixing a negative and a positive value breaks:

| float  | as signed int |
|--------|---------------|
| −0.1   | ≈ -1056964608 |
| +0.1   | ≈  1027384934 |

`atomicMax` correctly picks `+0.1`. But the initial value of `output` (host-allocated,
usually 0) is compared against the first block's negative result: `atomicMax(0, -1056964608)`
returns **0**, silently discarding the true maximum if all values are negative.

**Reproduction:** call `cuda_reduce_max` on an all-negative array; the returned maximum
will be `0.0f` instead of the (correct) largest negative value, because the host-side
`output` is initialised to 0 and `atomicMax` never overwrites it with a smaller int.

**Suggested fix:** either (a) initialise `output` to `__int_as_float(0xFF800000)` (−∞)
on the host before the kernel, or (b) run a second reduction kernel over the per-block
results using `fmaxf` instead of `atomicMax`.

---

## Summary

| # | File | Function | Severity | Symptom |
|---|------|----------|----------|---------|
| 1a | `search_space.py` | `estimate_complexity` | High | `UnboundLocalError` crash on `identity`-first genome |
| 1b | `search_space.py` | `estimate_complexity` | Medium | Params over-counted ×2 for `conv → identity` pairs |
| 1c | `search_space.py` | `estimate_complexity` | Medium | 1×1 conv params in `identity` silently omitted |
| 2a | `kernels.cu` | `reduceMax` | Medium | Returns 0 instead of true max when all inputs are negative |
