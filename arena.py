"""
Checkers Arena: tournament system with Elo ratings.

Pits different AI configurations against each other in round-robin
matches and ranks them by Elo rating. Useful for measuring whether
training actually produces stronger play, comparing hyperparameters,
and tracking improvement over time.

Usage:
    python arena.py                  # train configs + run tournament
    python arena.py --quick          # fewer games, faster results
    python arena.py --load results.json  # display saved results
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict

from checkers import CheckersGame
from ai import (
    Evaluator, DEFAULT_WEIGHTS, FEATURE_NAMES,
    choose_move, train_td, play_match,
)


@dataclass
class Player:
    """A named AI configuration."""
    name: str
    weights: list
    depth: int = 3
    description: str = ""

    def evaluator(self):
        return Evaluator(self.weights)


@dataclass
class MatchResult:
    """Result of a head-to-head match."""
    player1: str
    player2: str
    p1_wins: int
    p2_wins: int
    draws: int
    games: int


@dataclass
class TournamentResult:
    """Full tournament results."""
    players: list
    matches: list
    elo_ratings: dict
    records: dict  # name -> {wins, losses, draws}
    timestamp: str = ""


def expected_score(rating_a, rating_b):
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(ratings, player1, player2, result, k=32):
    """Update Elo ratings after a match.

    result: fraction of points player1 scored (wins + 0.5*draws) / games
    """
    e1 = expected_score(ratings[player1], ratings[player2])
    e2 = expected_score(ratings[player2], ratings[player1])

    ratings[player1] += k * (result - e1)
    ratings[player2] += k * ((1.0 - result) - e2)


def run_match(player1, player2, games_per_side=10):
    """Play a match between two players. Each side plays both colors."""
    total_games = games_per_side * 2
    ev1 = player1.evaluator()
    ev2 = player2.evaluator()

    p1_wins, p2_wins, draws = play_match(
        ev1, ev2, num_games=total_games,
        depth=max(player1.depth, player2.depth),
    )

    return MatchResult(
        player1=player1.name,
        player2=player2.name,
        p1_wins=p1_wins,
        p2_wins=p2_wins,
        draws=draws,
        games=total_games,
    )


def run_tournament(players, games_per_side=10, verbose=True):
    """Round-robin tournament. Every player plays every other player."""
    n = len(players)
    total_matches = n * (n - 1) // 2

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CHECKERS ARENA — {n} players, {total_matches} matches")
        print(f"  {games_per_side * 2} games per match ({games_per_side} per side)")
        print(f"{'='*60}\n")

    ratings = {p.name: 1500.0 for p in players}
    records = {p.name: {"wins": 0, "losses": 0, "draws": 0} for p in players}
    matches = []

    match_num = 0
    for i in range(n):
        for j in range(i + 1, n):
            match_num += 1
            p1, p2 = players[i], players[j]

            if verbose:
                print(f"  Match {match_num}/{total_matches}: "
                      f"{p1.name} vs {p2.name}", end=" ... ", flush=True)

            t0 = time.time()
            result = run_match(p1, p2, games_per_side)
            elapsed = time.time() - t0
            matches.append(result)

            # Update records
            records[p1.name]["wins"] += result.p1_wins
            records[p1.name]["losses"] += result.p2_wins
            records[p1.name]["draws"] += result.draws
            records[p2.name]["wins"] += result.p2_wins
            records[p2.name]["losses"] += result.p1_wins
            records[p2.name]["draws"] += result.draws

            # Update Elo (one update per match, based on score fraction)
            score = (result.p1_wins + 0.5 * result.draws) / result.games
            # Scale K by number of games for stability
            k = 32 * (result.games / 10)
            update_elo(ratings, p1.name, p2.name, score, k=k)

            if verbose:
                winner = (f"{p1.name} wins" if result.p1_wins > result.p2_wins
                          else f"{p2.name} wins" if result.p2_wins > result.p1_wins
                          else "Draw")
                print(f"{result.p1_wins}-{result.p2_wins}-{result.draws} "
                      f"({winner}) [{elapsed:.1f}s]")

    tournament = TournamentResult(
        players=[{"name": p.name, "weights": p.weights, "depth": p.depth,
                  "description": p.description} for p in players],
        matches=[asdict(m) for m in matches],
        elo_ratings={k: round(v, 1) for k, v in ratings.items()},
        records=records,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    if verbose:
        print_leaderboard(tournament)

    return tournament


def print_leaderboard(tournament):
    """Print a formatted leaderboard."""
    ratings = tournament.elo_ratings
    records = tournament.records
    ranked = sorted(ratings.items(), key=lambda x: -x[1])

    print(f"\n{'='*60}")
    print(f"  LEADERBOARD")
    print(f"{'='*60}")
    print(f"  {'#':<4}{'Player':<22}{'Elo':>6}  {'W':>4} {'L':>4} {'D':>4}  {'Win%':>5}")
    print(f"  {'-'*55}")

    for rank, (name, elo) in enumerate(ranked, 1):
        r = records[name]
        total = r["wins"] + r["losses"] + r["draws"]
        win_pct = (r["wins"] + 0.5 * r["draws"]) / total * 100 if total > 0 else 0
        print(f"  {rank:<4}{name:<22}{elo:>6.0f}  {r['wins']:>4} {r['losses']:>4} {r['draws']:>4}  {win_pct:>5.1f}%")

    print(f"{'='*60}")

    # Print match details
    print(f"\n  Match Results:")
    for m in tournament.matches:
        print(f"    {m['player1']:<20} vs {m['player2']:<20} "
              f"{m['p1_wins']}-{m['p2_wins']}-{m['draws']}")
    print()


def save_results(tournament, path="arena_results.json"):
    """Save tournament results to JSON."""
    with open(path, "w") as f:
        json.dump(asdict(tournament) if hasattr(tournament, '__dataclass_fields__')
                  else {"players": tournament.players, "matches": tournament.matches,
                        "elo_ratings": tournament.elo_ratings, "records": tournament.records,
                        "timestamp": tournament.timestamp},
                  f, indent=2)
    print(f"  Results saved to {path}")


def load_results(path="arena_results.json"):
    """Load and display saved tournament results."""
    with open(path) as f:
        data = json.load(f)
    result = TournamentResult(**data)
    print(f"\n  Tournament from {result.timestamp}")
    print_leaderboard(result)
    return result


def train_player(name, num_games, lr=0.003, depth=3, verbose=False):
    """Train a player configuration and return a Player."""
    if verbose:
        print(f"  Training '{name}' ({num_games} games, lr={lr})...", flush=True)
    t0 = time.time()
    evaluator, stats = train_td(num_games=num_games, lr=lr, depth=depth, verbose=False)
    elapsed = time.time() - t0
    if verbose:
        w = stats["black_wins"] + stats["white_wins"] - stats["draws"]
        print(f"    Done in {elapsed:.1f}s — "
              f"B:{stats['black_wins']} W:{stats['white_wins']} D:{stats['draws']}")
    return Player(
        name=name,
        weights=evaluator.weights[:],
        depth=depth,
        description=f"TD(λ) trained, {num_games} games, lr={lr}",
    )


def build_default_roster(quick=False):
    """Create a roster of players with different training configurations."""
    players = []

    # 1. Default (hand-tuned) weights
    players.append(Player(
        name="Default",
        weights=list(DEFAULT_WEIGHTS),
        description="Hand-tuned starting weights",
    ))

    # 2. Random (baseline)
    players.append(Player(
        name="Random",
        weights=[0.0] * len(DEFAULT_WEIGHTS),  # all zeros = random play
        depth=1,
        description="Random play (no evaluation)",
    ))

    # 3-5. Trained with different amounts of experience
    configs = [
        ("Rookie (100g)", 100),
        ("Trained (300g)", 300),
        ("Veteran (500g)", 500),
    ] if not quick else [
        ("Rookie (50g)", 50),
        ("Trained (150g)", 150),
    ]

    print(f"\n  Preparing players...")
    for name, num_games in configs:
        p = train_player(name, num_games, verbose=True)
        players.append(p)

    # 6. Aggressive — high piece weight, low defense
    players.append(Player(
        name="Aggressor",
        weights=[2.0, 2.5, 0.1, 0.3, 0.5, 0.3, 0.1, 0.1, 0.05, 0.05],
        description="Overvalues material, ignores defense",
    ))

    # 7. Turtle — high defense, low aggression
    players.append(Player(
        name="Turtle",
        weights=[0.8, 1.0, 1.5, 0.05, 0.1, -0.1, 0.02, 0.02, 0.5, 0.8],
        description="Defensive: back row + protection, avoids advancing",
    ))

    return players


def training_curve(total_games=600, checkpoint_every=50, match_games=20, depth=3):
    """Train incrementally and measure strength at each checkpoint.

    Shows how win rate against the Default baseline changes over
    training time. Reveals the point where additional training stops
    helping (or starts hurting).
    """
    from ai import CheckersGame, extract_features, NUM_FEATURES, TRAIN_WIN

    baseline = Player("Default", list(DEFAULT_WEIGHTS), depth=depth)
    evaluator = Evaluator()
    lr = 0.003
    current_lr = lr

    checkpoints = []
    num_checkpoints = total_games // checkpoint_every

    print(f"\n{'='*60}")
    print(f"  TRAINING CURVE — {total_games} games, checkpoint every {checkpoint_every}")
    print(f"  Measuring vs Default baseline ({match_games} games per checkpoint)")
    print(f"{'='*60}\n")

    phase1_end = int(total_games * 0.4)
    opponent = None
    stats = {"black_wins": 0, "white_wins": 0, "draws": 0}

    for game_num in range(total_games):
        # Phase transitions
        if game_num == phase1_end:
            opponent = Evaluator(evaluator.weights[:])
        elif game_num > phase1_end and game_num % 50 == 0:
            opponent = Evaluator(evaluator.weights[:])

        game = CheckersGame()
        positions = []
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
                    moves = game.get_legal_moves()
                    move = random.choice(moves)
                else:
                    move = choose_move(game, opponent, depth=depth)

            if move is None:
                break
            game.make_move(move)

        over, result = game.is_game_over()
        if result == learner_color:
            terminal_value = TRAIN_WIN
        elif result == -learner_color:
            terminal_value = -TRAIN_WIN
        else:
            terminal_value = 0.0

        if len(positions) >= 2:
            values = []
            for feats in positions:
                v = learner_color * sum(w * f for w, f in zip(evaluator.weights, feats))
                values.append(v)
            values.append(terminal_value)

            lam = 0.7
            trace = [0.0] * NUM_FEATURES
            for t in range(len(positions)):
                feats = positions[t]
                td_error = values[t + 1] - values[t]
                td_error = max(-0.5, min(0.5, td_error))
                for i in range(NUM_FEATURES):
                    trace[i] = lam * trace[i] + learner_color * feats[i]
                    evaluator.weights[i] += current_lr * td_error * trace[i]

            for i in range(NUM_FEATURES):
                evaluator.weights[i] = max(-10.0, min(10.0, evaluator.weights[i]))

        current_lr *= 0.999

        # Checkpoint?
        if (game_num + 1) % checkpoint_every == 0:
            cp_player = Player(
                f"G{game_num+1}",
                evaluator.weights[:],
                depth=depth,
            )
            # Measure vs baseline
            ev_cp = cp_player.evaluator()
            ev_base = baseline.evaluator()
            w, l, d = play_match(ev_cp, ev_base, num_games=match_games, depth=depth)
            win_rate = (w + 0.5 * d) / match_games * 100

            checkpoints.append({
                "games": game_num + 1,
                "wins": w,
                "losses": l,
                "draws": d,
                "win_rate": win_rate,
                "weights": evaluator.weights[:],
            })

            bar = "#" * int(win_rate / 2)
            print(f"  Game {game_num+1:>4}: {win_rate:5.1f}% vs Default  "
                  f"[{w}-{l}-{d}]  |{bar}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  TRAINING CURVE SUMMARY")
    print(f"{'='*60}")

    peak = max(checkpoints, key=lambda c: c["win_rate"])
    print(f"  Peak performance: {peak['win_rate']:.1f}% at game {peak['games']}")

    if checkpoints[-1]["win_rate"] < peak["win_rate"] - 5:
        drop = peak["win_rate"] - checkpoints[-1]["win_rate"]
        print(f"  Overfitting detected: -{drop:.1f}% from peak to final")
    else:
        print(f"  No significant overfitting detected")

    # Save curve data
    results_dir = os.path.dirname(os.path.abspath(__file__))
    curve_path = os.path.join(results_dir, "training_curve.json")
    with open(curve_path, "w") as f:
        json.dump({"checkpoints": checkpoints, "total_games": total_games,
                    "checkpoint_every": checkpoint_every}, f, indent=2)
    print(f"\n  Curve data saved to {curve_path}")

    return checkpoints


if __name__ == "__main__":
    quick = "--quick" in sys.argv

    if "--load" in sys.argv:
        idx = sys.argv.index("--load")
        path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "arena_results.json"
        load_results(path)
        sys.exit(0)

    if "--curve" in sys.argv:
        total = 300 if quick else 600
        every = 25 if quick else 50
        training_curve(total_games=total, checkpoint_every=every)
        sys.exit(0)

    games_per_side = 5 if quick else 10

    players = build_default_roster(quick=quick)
    tournament = run_tournament(players, games_per_side=games_per_side)

    results_dir = os.path.dirname(os.path.abspath(__file__))
    save_results(tournament, os.path.join(results_dir, "arena_results.json"))
