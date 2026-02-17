"""
Checkers AI: Arthur Samuel-style minimax with alpha-beta pruning
and a learned linear evaluation function trained via TD(0) self-play.
"""

import json
import math
import random
from checkers import CheckersGame, SQ_TO_RC, ADJACENT, JUMPS, RC_TO_SQ

# Feature names for readability
FEATURE_NAMES = [
    "piece_count",        # material advantage (men)
    "king_count",         # material advantage (kings)
    "back_row",           # pieces on back row (defensive)
    "center_men",         # men in center 8 squares
    "center_kings",       # kings in center 8 squares
    "advancement",        # how far forward men have pushed
    "mobility",           # number of legal moves
    "opp_mobility",       # opponent's legal moves (negated)
    "vulnerable",         # pieces under attack (negated)
    "protected",          # pieces with friendly support
]

NUM_FEATURES = len(FEATURE_NAMES)

# Normalization divisors — keep features roughly in [-1, 1]
# piece_count: max diff ±12, king_count: ±12, back_row: ±4, center: ±4,
# advancement: ~±40, mobility: ~±15, vulnerable: ±12, protected: ±12
FEATURE_SCALE = [12.0, 12.0, 4.0, 4.0, 4.0, 40.0, 15.0, 15.0, 12.0, 12.0]

# Center squares (rows 2-5, inner columns)
CENTER_SQUARES = set()
for sq in range(32):
    r, c = SQ_TO_RC[sq]
    if 2 <= r <= 5 and 2 <= c <= 5:
        CENTER_SQUARES.add(sq)

# Back row squares
BLACK_BACK_ROW = {sq for sq in range(32) if SQ_TO_RC[sq][0] == 0}
WHITE_BACK_ROW = {sq for sq in range(32) if SQ_TO_RC[sq][0] == 7}

DEFAULT_WEIGHTS = [1.0, 1.5, 0.3, 0.1, 0.3, 0.1, 0.05, 0.05, 0.15, 0.1]

# Terminal values for search (large) vs training (small)
SEARCH_WIN = 1000.0
TRAIN_WIN = 1.0


def extract_features(game):
    """Extract 10 Samuel-style features from black's perspective, normalized."""
    board = game.board
    feats = [0.0] * NUM_FEATURES

    for sq in range(32):
        piece = board[sq]
        if piece == 0:
            continue

        r, _ = SQ_TO_RC[sq]

        if piece == 1:  # black man
            feats[0] += 1
            if sq in CENTER_SQUARES:
                feats[3] += 1
            if sq in BLACK_BACK_ROW:
                feats[2] += 1
            feats[5] += r
        elif piece == 2:  # black king
            feats[1] += 1
            if sq in CENTER_SQUARES:
                feats[4] += 1
        elif piece == -1:  # white man
            feats[0] -= 1
            if sq in CENTER_SQUARES:
                feats[3] -= 1
            if sq in WHITE_BACK_ROW:
                feats[2] -= 1
            feats[5] -= (7 - r)
        elif piece == -2:  # white king
            feats[1] -= 1
            if sq in CENTER_SQUARES:
                feats[4] -= 1

    # Mobility
    current_moves = len(game.get_legal_moves())
    game.turn *= -1
    opp_moves = len(game.get_legal_moves())
    game.turn *= -1

    if game.turn == 1:
        feats[6] = current_moves
        feats[7] = -opp_moves
    else:
        feats[6] = opp_moves
        feats[7] = -current_moves

    # Vulnerable and protected pieces
    for sq in range(32):
        piece = board[sq]
        if piece == 0:
            continue
        is_black = piece > 0
        sign = 1 if is_black else -1

        vulnerable = False
        has_friendly = False
        for (dr, dc), nsq in ADJACENT[sq]:
            neighbor = board[nsq]
            if neighbor == 0:
                continue
            neighbor_is_black = neighbor > 0
            if neighbor_is_black == is_black:
                has_friendly = True
            else:
                land_r = SQ_TO_RC[sq][0] + (SQ_TO_RC[sq][0] - SQ_TO_RC[nsq][0])
                land_c = SQ_TO_RC[sq][1] + (SQ_TO_RC[sq][1] - SQ_TO_RC[nsq][1])
                if 0 <= land_r < 8 and 0 <= land_c < 8:
                    land_sq = RC_TO_SQ.get((land_r, land_c))
                    if land_sq is not None and board[land_sq] == 0:
                        vulnerable = True

        if vulnerable:
            feats[8] -= sign
        if has_friendly:
            feats[9] += sign

    # Normalize features
    for i in range(NUM_FEATURES):
        feats[i] /= FEATURE_SCALE[i]

    return feats


class Evaluator:
    """Linear evaluation function: score = dot(weights, features)."""

    def __init__(self, weights=None):
        self.weights = list(weights) if weights else list(DEFAULT_WEIGHTS)

    def evaluate(self, game):
        """Return evaluation score from black's perspective."""
        over, result = game.is_game_over()
        if over:
            if result == 1:
                return SEARCH_WIN
            elif result == -1:
                return -SEARCH_WIN
            return 0.0

        feats = extract_features(game)
        return sum(w * f for w, f in zip(self.weights, feats))

    def save(self, path="weights.json"):
        with open(path, "w") as f:
            json.dump({"weights": self.weights, "feature_names": FEATURE_NAMES}, f, indent=2)

    @classmethod
    def load(cls, path="weights.json"):
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(data["weights"])
        except FileNotFoundError:
            return cls()


def minimax(game, depth, alpha, beta, evaluator, maximizing):
    """Minimax with alpha-beta pruning.

    maximizing=True means it's black's turn (maximize score).
    Returns (score, best_move).
    """
    over, result = game.is_game_over()
    if over:
        if result == 1:
            return SEARCH_WIN, None
        elif result == -1:
            return -SEARCH_WIN, None
        return 0.0, None

    if depth == 0:
        return evaluator.evaluate(game), None

    moves = game.get_legal_moves()
    # Move ordering: captures first (longer moves = multi-jumps) for better pruning
    moves.sort(key=lambda m: -len(m))

    best_move = moves[0]

    if maximizing:
        max_eval = -math.inf
        for move in moves:
            child = game.copy()
            child.make_move(move)
            score, _ = minimax(child, depth - 1, alpha, beta, evaluator, False)
            if score > max_eval:
                max_eval = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            child = game.copy()
            child.make_move(move)
            score, _ = minimax(child, depth - 1, alpha, beta, evaluator, True)
            if score < min_eval:
                min_eval = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:
                break
        return min_eval, best_move


def choose_move(game, evaluator, depth=5):
    """Choose the best move for the current player."""
    maximizing = game.turn == 1
    _, move = minimax(game, depth, -math.inf, math.inf, evaluator, maximizing)
    return move


def train_td(num_games=1000, lr=0.003, lr_decay=0.999, depth=3, verbose=True):
    """Train weights via TD(lambda) self-play.

    Phase 1 (first 40%): train against random opponent to learn basics.
    Phase 2 (remaining 60%): self-play against frozen copy for refinement.
    The learner alternates colors each game. TD updates only apply to
    the learner's positions.
    """
    evaluator = Evaluator()
    current_lr = lr
    phase1_end = int(num_games * 0.4)

    stats = {"black_wins": 0, "white_wins": 0, "draws": 0}
    opponent = None  # None = random, Evaluator = frozen copy

    for game_num in range(num_games):
        # Phase transitions
        if game_num == phase1_end:
            opponent = Evaluator(evaluator.weights[:])
            if verbose:
                print("  >> Switching to self-play phase")
        elif game_num > phase1_end and game_num % 50 == 0:
            opponent = Evaluator(evaluator.weights[:])

        game = CheckersGame()
        positions = []

        # Alternate which color the learner plays
        learner_color = 1 if game_num % 2 == 0 else -1

        for _ in range(300):
            over, result = game.is_game_over()
            if over:
                break

            if game.turn == learner_color:
                feats = extract_features(game)
                positions.append(feats)
                if random.random() < 0.1:
                    moves = game.get_legal_moves()
                    move = random.choice(moves)
                else:
                    move = choose_move(game, evaluator, depth=depth)
            else:
                if opponent is None:
                    # Phase 1: random opponent
                    moves = game.get_legal_moves()
                    move = random.choice(moves)
                else:
                    move = choose_move(game, opponent, depth=depth)

            if move is None:
                break
            game.make_move(move)

        over, result = game.is_game_over()
        # Terminal value from learner's perspective
        if result == learner_color:
            terminal_value = TRAIN_WIN
            stats["black_wins" if learner_color == 1 else "white_wins"] += 1
        elif result == -learner_color:
            terminal_value = -TRAIN_WIN
            stats["white_wins" if learner_color == 1 else "black_wins"] += 1
        else:
            terminal_value = 0.0
            stats["draws"] += 1

        if len(positions) < 2:
            continue

        # Features are always from black's POV. The evaluator's score = dot(w, f)
        # is from black's POV. For learner=white, a good position has negative eval.
        # We want V(s) = learner_color * dot(w, f) so that V > 0 means good for learner.
        values = []
        for feats in positions:
            v = learner_color * sum(w * f for w, f in zip(evaluator.weights, feats))
            values.append(v)
        values.append(terminal_value)

        # TD(lambda) updates with eligibility traces
        lam = 0.7
        trace = [0.0] * NUM_FEATURES
        for t in range(len(positions)):
            feats = positions[t]
            td_error = values[t + 1] - values[t]
            td_error = max(-0.5, min(0.5, td_error))

            # Gradient of V(s) w.r.t. weights is learner_color * feats
            for i in range(NUM_FEATURES):
                trace[i] = lam * trace[i] + learner_color * feats[i]
                evaluator.weights[i] += current_lr * td_error * trace[i]

        # Clip weights
        for i in range(NUM_FEATURES):
            evaluator.weights[i] = max(-10.0, min(10.0, evaluator.weights[i]))

        current_lr *= lr_decay

        if verbose and (game_num + 1) % 100 == 0:
            total = game_num + 1
            print(f"Game {total}/{num_games}: "
                  f"B={stats['black_wins']} W={stats['white_wins']} D={stats['draws']} "
                  f"LR={current_lr:.6f}")
            print(f"  Weights: {[f'{w:.3f}' for w in evaluator.weights]}")

    evaluator.save()
    if verbose:
        print(f"\nTraining complete. Final weights saved.")
        for name, w in zip(FEATURE_NAMES, evaluator.weights):
            print(f"  {name}: {w:.4f}")

    return evaluator, stats


def play_match(evaluator1, evaluator2, num_games=100, depth=5):
    """Play a match between two evaluators. Returns (e1_wins, e2_wins, draws)."""
    e1_wins = e2_wins = draws = 0

    for i in range(num_games):
        game = CheckersGame()
        e1_is_black = (i % 2 == 0)

        for _ in range(300):
            over, result = game.is_game_over()
            if over:
                break
            if game.turn == 1:
                ev = evaluator1 if e1_is_black else evaluator2
            else:
                ev = evaluator2 if e1_is_black else evaluator1
            move = choose_move(game, ev, depth=depth)
            if move is None:
                break
            game.make_move(move)

        over, result = game.is_game_over()
        if result == 0:
            draws += 1
        elif (result == 1 and e1_is_black) or (result == -1 and not e1_is_black):
            e1_wins += 1
        else:
            e2_wins += 1

    return e1_wins, e2_wins, draws


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        train_td(num_games=n)
    else:
        print("Training 200 games (depth 3)...")
        evaluator, stats = train_td(num_games=200, verbose=True)

        print("\nTrained vs default weights (20 games, depth 3):")
        default_ev = Evaluator()
        w, l, d = play_match(evaluator, default_ev, num_games=20, depth=3)
        print(f"  Trained wins: {w}, Default wins: {l}, Draws: {d}")
