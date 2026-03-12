# vla-game-agent

A tiny instruction-conditioned vision-to-action agent for [Crafter](https://github.com/danijar/crafter).

Takes a game frame + a text instruction → predicts a player action.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python scripts/random_rollout.py --seed 42 --episodes 3 --render
```

## Tests

```bash
uv run python -m pytest
```

## License

MIT
