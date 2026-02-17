"""
Checkers game engine.

Board representation: 32 playable (dark) squares, numbered 0-31.
Standard checkers layout (rows top to bottom, left to right):

    Row 0:  [ ][0][ ][1][ ][2][ ][3]
    Row 1:  [4][ ][5][ ][6][ ][7][ ]
    Row 2:  [ ][8][ ][9][ ][10][ ][11]
    Row 3:  [12][ ][13][ ][14][ ][15][ ]
    Row 4:  [ ][16][ ][17][ ][18][ ][19]
    Row 5:  [20][ ][21][ ][22][ ][23][ ]
    Row 6:  [ ][24][ ][25][ ][26][ ][27]
    Row 7:  [28][ ][29][ ][30][ ][31][ ]

Piece encoding:
    0  = empty
    1  = black man
    2  = black king
    -1 = white man
    -2 = white king

Black moves DOWN (toward row 7), white moves UP (toward row 0).
Black moves first.
"""

import copy

# Square index -> (row, col) on the 8x8 board
SQ_TO_RC = {}
RC_TO_SQ = {}
for sq in range(32):
    r = sq // 4
    # Even rows: dark squares at cols 1,3,5,7
    # Odd rows:  dark squares at cols 0,2,4,6
    c = 2 * (sq % 4) + (1 if r % 2 == 0 else 0)
    SQ_TO_RC[sq] = (r, c)
    RC_TO_SQ[(r, c)] = sq

# Precomputed adjacency tables
# For each square, list of (neighbor, jumped_over) pairs for each direction
# Directions: forward-left, forward-right (relative to piece color) handled at move time
# Here we store absolute geometric adjacency.

# Simple move neighbors (diagonal adjacency)
ADJACENT = [[] for _ in range(32)]  # sq -> list of (direction, neighbor_sq)
# Jump neighbors
JUMPS = [[] for _ in range(32)]     # sq -> list of (direction, over_sq, land_sq)

# Directions: (dr, dc) — all four diagonals
_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

for sq in range(32):
    r, c = SQ_TO_RC[sq]
    for dr, dc in _DIRS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 8 and 0 <= nc < 8:
            nsq = RC_TO_SQ.get((nr, nc))
            if nsq is not None:
                ADJACENT[sq].append(((dr, dc), nsq))
        # Jump
        jr, jc = r + 2 * dr, c + 2 * dc
        if 0 <= jr < 8 and 0 <= jc < 8:
            over = RC_TO_SQ.get((nr, nc))
            land = RC_TO_SQ.get((jr, jc))
            if over is not None and land is not None:
                JUMPS[sq].append(((dr, dc), over, land))

# Rows for promotion
BLACK_KING_ROW = 7  # black promotes on row 7
WHITE_KING_ROW = 0  # white promotes on row 0

# Draw threshold
DRAW_HALF_MOVES = 80


class CheckersGame:
    """Full checkers game state with move generation and game logic."""

    def __init__(self):
        self.board = [0] * 32
        # Initial setup: black on rows 0-2 (squares 0-11), white on rows 5-7 (squares 20-31)
        for sq in range(12):
            self.board[sq] = 1   # black man
        for sq in range(20, 32):
            self.board[sq] = -1  # white man
        self.turn = 1  # 1 = black, -1 = white
        self.half_moves_since_capture = 0
        self.move_history = []

    def copy(self):
        g = CheckersGame.__new__(CheckersGame)
        g.board = self.board[:]
        g.turn = self.turn
        g.half_moves_since_capture = self.half_moves_since_capture
        g.move_history = self.move_history[:]
        return g

    def _forward_dirs(self, color):
        """Return allowed row-directions for a man of the given color."""
        if color == 1:  # black moves down
            return [1]
        else:           # white moves up
            return [-1]

    def _get_simple_moves(self):
        """Generate all non-capture moves for the current player."""
        moves = []
        for sq in range(32):
            piece = self.board[sq]
            if piece == 0 or (piece > 0) != (self.turn > 0):
                continue
            is_king = abs(piece) == 2
            allowed_dr = self._forward_dirs(self.turn) if not is_king else [-1, 1]
            for (dr, dc), nsq in ADJACENT[sq]:
                if dr not in allowed_dr:
                    continue
                if self.board[nsq] == 0:
                    moves.append([sq, nsq])
        return moves

    def _get_jump_sequences(self):
        """Generate all capture sequences (multi-jumps) via DFS."""
        all_jumps = []
        for sq in range(32):
            piece = self.board[sq]
            if piece == 0 or (piece > 0) != (self.turn > 0):
                continue
            # DFS for jump chains from this square
            self._dfs_jumps(sq, piece, self.board[:], [sq], set(), all_jumps)
        return all_jumps

    def _dfs_jumps(self, sq, piece, board, path, captured, all_jumps):
        """DFS to find all maximal jump sequences from sq."""
        is_king = abs(piece) == 2
        allowed_dr = self._forward_dirs(self.turn) if not is_king else [-1, 1]
        found_jump = False

        for (dr, dc), over, land in JUMPS[sq]:
            if dr not in allowed_dr:
                continue
            if over in captured:
                continue
            over_piece = board[over]
            # Must jump over opponent piece
            if over_piece == 0 or (over_piece > 0) == (self.turn > 0):
                continue
            if board[land] != 0:
                continue

            # Check if landing promotes this piece
            land_row = SQ_TO_RC[land][0]
            promotes = False
            if abs(piece) == 1:
                if self.turn == 1 and land_row == BLACK_KING_ROW:
                    promotes = True
                elif self.turn == -1 and land_row == WHITE_KING_ROW:
                    promotes = True

            found_jump = True

            # Execute jump on temp board
            board[sq] = 0
            board[land] = piece
            board[over] = 0
            captured.add(over)
            path.append(land)

            if promotes:
                # Promotion ends the turn — no further jumps
                all_jumps.append(list(path))
            else:
                self._dfs_jumps(land, piece, board, path, captured, all_jumps)

            # Undo
            path.pop()
            captured.discard(over)
            board[sq] = piece
            board[land] = 0
            board[over] = over_piece

        if not found_jump and len(path) > 1:
            all_jumps.append(list(path))

    def get_legal_moves(self):
        """Return list of legal moves. Each move is a list of square indices [from, ..., to].
        If jumps exist, only jumps are returned (mandatory capture)."""
        jumps = self._get_jump_sequences()
        if jumps:
            return jumps
        return self._get_simple_moves()

    def make_move(self, move):
        """Execute a move (list of squares). Returns self for chaining."""
        if len(move) < 2:
            raise ValueError("Move must have at least 2 squares")

        start = move[0]
        piece = self.board[start]
        is_capture = False

        for i in range(len(move) - 1):
            fr, to = move[i], move[i + 1]
            self.board[fr] = 0

            # If it's a jump, remove the captured piece
            fr_r, fr_c = SQ_TO_RC[fr]
            to_r, to_c = SQ_TO_RC[to]
            if abs(to_r - fr_r) == 2:
                mid_r, mid_c = (fr_r + to_r) // 2, (fr_c + to_c) // 2
                mid_sq = RC_TO_SQ[(mid_r, mid_c)]
                self.board[mid_sq] = 0
                is_capture = True

            self.board[to] = piece

        end = move[-1]
        end_row = SQ_TO_RC[end][0]

        # King promotion
        if piece == 1 and end_row == BLACK_KING_ROW:
            self.board[end] = 2
        elif piece == -1 and end_row == WHITE_KING_ROW:
            self.board[end] = -2

        if is_capture:
            self.half_moves_since_capture = 0
        else:
            self.half_moves_since_capture += 1

        self.move_history.append(move)
        self.turn *= -1
        return self

    def is_game_over(self):
        """Return (is_over, result) where result is 1 (black wins), -1 (white wins), or 0 (draw)."""
        if self.half_moves_since_capture >= DRAW_HALF_MOVES:
            return True, 0
        if not self.get_legal_moves():
            return True, -self.turn  # current player loses
        return False, None

    def winner_name(self):
        over, result = self.is_game_over()
        if not over:
            return None
        if result == 1:
            return "black"
        elif result == -1:
            return "white"
        return "draw"

    def to_dict(self):
        """Serialize game state for the frontend."""
        over, result = self.is_game_over()
        return {
            "board": self.board[:],
            "turn": self.turn,
            "is_over": over,
            "result": result,
            "move_count": len(self.move_history),
        }

    def to_8x8(self):
        """Convert to 8x8 grid for display (0=empty, piece codes on dark squares, None on light)."""
        grid = [[None] * 8 for _ in range(8)]
        for sq in range(32):
            r, c = SQ_TO_RC[sq]
            grid[r][c] = self.board[sq]
        return grid

    def __repr__(self):
        symbols = {0: ".", 1: "b", 2: "B", -1: "w", -2: "W"}
        grid = self.to_8x8()
        lines = []
        for r in range(8):
            row_str = ""
            for c in range(8):
                v = grid[r][c]
                row_str += symbols.get(v, " ") if v is not None else " "
            lines.append(row_str)
        turn_name = "Black" if self.turn == 1 else "White"
        lines.append(f"Turn: {turn_name}")
        return "\n".join(lines)
