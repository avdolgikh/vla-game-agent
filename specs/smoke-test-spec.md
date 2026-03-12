# Spec: Smoke Test (Pipeline Validation)

## Status

Approved

## Goal

Validate the agentic TDD pipeline works end-to-end with a trivial module.

## Scope

Create a single module `src/vla_agent/utils/seed.py` with one function.

## Requirements

### `set_seed(seed: int) -> None`

- Sets the random seed for Python's `random` module and `numpy`.
- Accepts any non-negative integer.
- Raises `ValueError` if seed is negative.

### `get_rng(seed: int) -> numpy.random.Generator`

- Returns a `numpy.random.Generator` seeded with the given value.
- Two calls with the same seed must produce generators that yield identical sequences.
- Raises `ValueError` if seed is negative.

## Acceptance Criteria

### AC-1: set_seed reproducibility

- After `set_seed(42)`, `random.random()` returns the same value on repeated calls with the same seed.

### AC-2: get_rng reproducibility

- `get_rng(42).random()` returns the same float every time.

### AC-3: Different seeds differ

- `get_rng(0).random() != get_rng(1).random()`

### AC-4: Negative seed rejected

- `set_seed(-1)` raises `ValueError`.
- `get_rng(-1)` raises `ValueError`.

## Package Layout

```
src/vla_agent/
  __init__.py
  utils/
    __init__.py
    seed.py
```
