"""
Flask server for checkers game.
Human plays black (moves first), AI plays white.
"""

import os
from flask import Flask, jsonify, request, render_template
from checkers import CheckersGame, SQ_TO_RC
from ai import Evaluator, choose_move, train_td, FEATURE_NAMES

app = Flask(__name__)

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights.json")

# Global game state and AI
game = CheckersGame()
evaluator = Evaluator.load(WEIGHTS_PATH)


def reset_game():
    global game
    game = CheckersGame()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    return jsonify(game.to_dict())


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    reset_game()
    return jsonify(game.to_dict())


@app.route("/api/legal_moves", methods=["GET"])
def api_legal_moves():
    """Return legal moves for the current player.
    Each move is a list of square indices [from, ..., to]."""
    moves = game.get_legal_moves()
    # Group by source square for the UI
    by_source = {}
    for move in moves:
        src = move[0]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(move)
    return jsonify({"moves": moves, "by_source": by_source})


@app.route("/api/make_move", methods=["POST"])
def api_make_move():
    """Human makes a move. Returns updated state (does NOT trigger AI)."""
    data = request.json
    move = data.get("move")

    if not move:
        return jsonify({"error": "No move provided"}), 400

    if game.turn != 1:
        return jsonify({"error": "Not your turn"}), 400

    legal = game.get_legal_moves()
    if move not in legal:
        return jsonify({"error": "Illegal move"}), 400

    game.make_move(move)
    return jsonify(game.to_dict())


@app.route("/api/ai_move", methods=["POST"])
def api_ai_move():
    """AI makes a move. Returns its move and updated state."""
    if game.turn != -1:
        return jsonify({"error": "Not AI's turn"}), 400

    over, _ = game.is_game_over()
    if over:
        return jsonify({"state": game.to_dict(), "ai_move": None})

    ai_move = choose_move(game, evaluator, depth=5)
    if ai_move:
        game.make_move(ai_move)

    return jsonify({"state": game.to_dict(), "ai_move": ai_move})


@app.route("/api/train", methods=["POST"])
def api_train():
    """Run self-play training."""
    global evaluator
    data = request.json or {}
    num_games = min(data.get("num_games", 500), 2000)
    depth = data.get("depth", 3)

    evaluator, stats = train_td(num_games=num_games, depth=depth, verbose=False)
    evaluator.save(WEIGHTS_PATH)

    return jsonify({
        "stats": stats,
        "weights": dict(zip(
            [f.replace("_", " ") for f in ["piece_count", "king_count", "back_row",
             "center_men", "center_kings", "advancement", "mobility",
             "opp_mobility", "vulnerable", "protected"]],
            [round(w, 4) for w in evaluator.weights]
        ))
    })


@app.route("/api/arena/results")
def api_arena_results():
    """Return saved arena tournament results."""
    results_path = os.path.join(os.path.dirname(__file__), "arena_results.json")
    try:
        with open(results_path) as f:
            import json
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"error": "No arena results yet. Run: python arena.py"}), 404


if __name__ == "__main__":
    app.run(port=5050, debug=True)
