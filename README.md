# Checkers AI — Arthur Samuel Style

A checkers game with an AI opponent that learns to play through self-play, inspired by Arthur Samuel's 1959 checkers program — one of the earliest examples of machine learning.

## Quick Start

```bash
pip install flask
python app.py
```

Open http://localhost:5050. You play black (first move). Click a piece, click a highlighted square, and the AI responds.

To train a stronger AI from scratch:

```bash
python ai.py train 1000
```

## How It Works

### The Game Engine

The board uses the standard 32-square representation — only the dark diagonal squares are playable. Squares are numbered 0-31, top-left to bottom-right. Precomputed adjacency tables make move generation fast: for each square, we know exactly which squares are reachable by a simple move or a jump, without doing any coordinate math at runtime.

The engine enforces mandatory captures (if you can jump, you must), multi-jump chains via DFS, and king promotion on the back row.

### The Evaluation Function

The AI evaluates positions using 10 hand-picked features, all computed from black's perspective and normalized to roughly [-1, 1]:

| Feature | What it measures |
|---------|-----------------|
| piece_count | Material advantage in men |
| king_count | Material advantage in kings |
| back_row | Pieces defending the back row |
| center_men | Men controlling the center 4x4 |
| center_kings | Kings controlling the center |
| advancement | How far forward men have pushed |
| mobility | Number of available moves |
| opp_mobility | Opponent's available moves (negated) |
| vulnerable | Pieces that can be captured (negated) |
| protected | Pieces with friendly neighbors |

The position score is a simple dot product: `score = weights . features`. The entire "brain" of the AI is just 10 numbers.

### The Search

Minimax with alpha-beta pruning, searching to depth 5 during play. Move ordering (captures first, multi-jumps first) improves pruning efficiency. At depth 5, the AI typically evaluates thousands of positions per move.

### The Training — TD(lambda) Self-Play

This is the Arthur Samuel part. The AI learns its 10 weights through temporal difference learning:

**Phase 1 — Learning the basics (first 40% of games):** The AI plays against a random opponent. This provides clear signal — winning against random play teaches that more pieces is good, kings are valuable, and getting captured is bad. The learner alternates playing black and white each game.

**Phase 2 — Self-improvement (remaining 60%):** The AI plays against a frozen copy of itself. Every 50 games, the frozen opponent is updated to the current weights. This creates an arms race where the AI must find improvements over its previous self.

The TD(lambda) update rule with lambda=0.7:

```
For each position the learner visited:
    td_error = V(next_position) - V(current_position)
    eligibility_trace = 0.7 * trace + gradient
    weights += learning_rate * td_error * eligibility_trace
```

The eligibility trace (lambda=0.7) is key — it propagates the game outcome backward through the entire sequence of positions, not just one step at a time. A game-ending blunder gets "blamed" on the positions that led to it, even many moves earlier.

Exploration: 10% of moves are random (epsilon-greedy), ensuring the AI doesn't get stuck in a narrow repertoire.

### What Went Wrong Along the Way

Getting TD learning to work required solving several instabilities:

**Weight explosion.** The first attempt used raw (unnormalized) features and large terminal values (+-1000). Feature values like mobility (~7) and advancement (~20) multiplied by large TD errors caused weights to grow to 10^80 within 100 games. Fix: normalize all features to [-1, 1] and use +-1.0 terminal values for training (keeping +-1000 only for search).

**Sign confusion.** Features are always computed from black's perspective. When the learner plays white, the value function must be negated (`V = -1 * dot(w, f)`) and the gradient must also flip sign. Getting this wrong caused the mobility weight to go strongly negative — the AI actively avoided having moves available.

**Defensive overfitting.** Training against random opponents taught the AI that keeping pieces on the back row is extremely important (weight went from 0.3 to 0.95). Against random play, back-row pieces do survive longer. But against a competent opponent, this excessive caution prevents advancing pieces and loses games. Fix: lower learning rate (0.003) and the two-phase approach, where self-play in Phase 2 tempers the lessons from Phase 1.

## What the AI Learned

After 300 games of training, the learned weights vs initial defaults:

| Feature | Default | Learned | What it means |
|---------|---------|---------|---------------|
| piece_count | 1.00 | **1.12** | Material matters even more than expected |
| king_count | 1.50 | **1.57** | Kings slightly more valuable than 1.5x a man |
| back_row | 0.30 | **0.66** | Defense is important (biggest relative change) |
| center_men | 0.10 | **0.21** | Center control matters for men |
| center_kings | 0.30 | **0.37** | Center kings are strong |
| advancement | 0.10 | **0.11** | Slight push forward is good |
| mobility | 0.05 | **0.04** | Having options is mildly good |
| opp_mobility | 0.05 | **0.03** | Restricting opponent is mildly good |
| vulnerable | 0.15 | **0.19** | Avoid being exposed to captures |
| protected | 0.10 | **0.18** | Keep pieces close together |

The biggest lesson the AI learned: **defense wins games.** The back_row weight more than doubled, and the protection weight nearly doubled. This aligns with real checkers strategy — maintaining your back row prevents king invasions, and keeping pieces connected prevents them from being picked off.

The AI also learned that kings are slightly more valuable than the naive 1.5x multiplier, and that center control with kings (0.37) matters more than center control with men (0.21).

## Results

| Match | Score |
|-------|-------|
| Trained vs Random (depth 5) | **20-0** |
| Trained vs Untrained (depth 3) | **5-0** (5 draws) |
| Trained vs Random (depth 3) | **20-0** |

The draws in trained-vs-untrained come from color alternation — at depth 3, one color has a systematic advantage in certain lines. In all games where the trained AI doesn't draw, it wins.

## Arena — Tournament System

The arena pits different AI configurations against each other in round-robin matches with Elo ratings. This answers the key question: **does training actually produce a stronger player?**

### Running a Tournament

```bash
python arena.py              # full tournament (trains 3 configs + 2 hand-crafted)
python arena.py --quick      # fewer training games, faster results
python arena.py --load arena_results.json  # display saved results
```

The default roster includes:
- **Default** — hand-tuned starting weights
- **Random** — no evaluation (baseline)
- **Rookie** — TD(lambda) trained for 100 games
- **Trained** — TD(lambda) trained for 300 games
- **Veteran** — TD(lambda) trained for 500 games
- **Aggressor** — hand-crafted weights that overvalue material
- **Turtle** — hand-crafted weights that prioritize defense

Each pair plays 20 games (10 per side) with Elo updates after each match.

### Viewing Results

Results are saved to `arena_results.json` and viewable:
- **CLI**: `python arena.py --load arena_results.json`
- **Web UI**: Click the "Arena" button in the game interface

### What the Arena Reveals

Training produces measurably stronger play. The Elo gap between trained and untrained configurations is significant and consistent across tournament runs. Hand-crafted extreme strategies (pure aggression, pure defense) perform worse than balanced learned weights — the TD(lambda) learner finds a better trade-off than any human-designed heuristic we tried.

## Architecture

```
checkers.py    284 lines   Game engine, board, moves, rules
ai.py          385 lines   Features, evaluator, minimax, training
arena.py       220 lines   Tournament engine with Elo ratings
app.py         125 lines   Flask API server
index.html     480 lines   Board UI + arena leaderboard
weights.json    26 lines   Learned weights
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/state` | GET | Current board state |
| `/api/new_game` | POST | Reset the board |
| `/api/legal_moves` | GET | Legal moves for current player |
| `/api/make_move` | POST | Human moves, AI responds |
| `/api/train` | POST | Run self-play training (blocking) |
| `/api/arena/results` | GET | Latest tournament leaderboard |

## References

- Samuel, A. L. (1959). "Some Studies in Machine Learning Using the Game of Checkers." *IBM Journal of Research and Development*, 3(3), 210-229.
- Sutton, R. S. (1988). "Learning to Predict by the Methods of Temporal Differences." *Machine Learning*, 3(1), 9-44.
