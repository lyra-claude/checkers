# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A checkers game with an AI opponent that learns through TD(lambda) self-play, inspired by Arthur Samuel's 1959 program. Flask backend, vanilla JS/CSS frontend (single-page, no build step).

## Commands

```bash
# Run the web app (serves on http://localhost:5050)
python app.py

# Train AI from CLI (default 1000 games, also runs a 20-game match vs defaults)
python ai.py

# Train with specific game count
python ai.py train 500
```

Only dependency is Flask (`pip install flask`).

## Architecture

Three-file backend, one-file frontend:

- **`checkers.py`** — Game engine. Board is 32-element array (only dark squares). Pieces: `1`/`2` = black man/king, `-1`/`-2` = white. Precomputed `ADJACENT` and `JUMPS` tables for move generation. Multi-jump captures via DFS in `_dfs_jumps`. Mandatory capture rule enforced in `get_legal_moves`.

- **`ai.py`** — AI brain. `extract_features()` computes 10 hand-crafted features always from black's perspective, normalized to [-1, 1]. `Evaluator` is a linear model (dot product of weights and features). Search is minimax with alpha-beta pruning. `train_td()` does two-phase learning: 40% vs random, then 60% self-play with frozen opponent refreshed every 50 games. TD error clamped to [-0.5, 0.5], weights clamped to [-10, 10].

- **`app.py`** — Flask API. Global `game` and `evaluator` state. Human is always black (turn=1), AI is always white (turn=-1). AI search depth is 5 during play, 3 during training. The `/api/train` endpoint is blocking (can take minutes).

- **`templates/index.html`** — Full UI in one file (inline JS/CSS). Supports both click-to-move and drag-and-drop. Board rendered as an 8x8 CSS grid, squares mapped via `rc2sq`/`sq2rc` matching the backend's `SQ_TO_RC`/`RC_TO_SQ`.

- **`weights.json`** — Learned weights, loaded at server start. Updated by training (both CLI and API).

## Key Design Decisions

- All features are computed from black's perspective. When the learner plays white during training, the value function is negated (`V = learner_color * dot(w, f)`) and the gradient flips sign accordingly.
- Terminal values differ by context: `+-1000` for search (to make wins/losses dominate), `+-1.0` for TD training (to keep updates stable).
- Move ordering sorts by move length descending (multi-jumps first) to improve alpha-beta pruning.
- The `/api/make_move` and `/api/ai_move` are separate endpoints — the frontend calls make_move, then ai_move sequentially with a 600ms delay for visual pacing.
