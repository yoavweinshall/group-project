"""
Core AI routines for ChessLab.

All the student work for this assignment lives in this file. Implement the
functions below so that:
  * `evaluate` returns a heuristic score for the given board state.
  * `choose_minimax_move` picks a move via minimax search (no pruning).
  * `choose_alphabeta_move` picks a move via minimax with alpha-beta pruning.

The helper `choose_random_move` is provided for you.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Set

from ..board import Board, Move
from ..common.profiling import Counter

MoveType = Tuple[Tuple[int, int], Tuple[int, int], Optional[str]]
PIECES_SCORE = {'P':100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
PIECES_LOCATION_SCORE = {
                'P':[
                        [0,  0,  0,  0,  0,  0,  0,  0],
                        [50, 50, 50, 50, 50, 50, 50, 50],
                        [10, 10, 20, 30, 30, 20, 10, 10],
                        [5,  5, 10, 25, 25, 10,  5,  5],
                        [0,  0,  0, 20, 20,  0,  0,  0],
                        [5, -5,-10,  0,  0,-10, -5,  5],
                        [5, 10, 10,-20,-20, 10, 10,  5],
                        [0,  0,  0,  0,  0,  0,  0,  0]
                    ],
                'N': [
                        [-50,-40,-30,-30,-30,-30,-40,-50],
                        [-40,-20,  0,  0,  0,  0,-20,-40],
                        [-30,  0, 10, 15, 15, 10,  0,-30],
                        [-30,  5, 15, 20, 20, 15,  5,-30],
                        [-30,  0, 15, 20, 20, 15,  0,-30],
                        [-30,  5, 10, 15, 15, 10,  5,-30],
                        [-40,-20,  0,  5,  5,  0,-20,-40],
                        [-50,-40,-30,-30,-30,-30,-40,-50]
                    ],
                'B': [
                        [-20,-10,-10,-10,-10,-10,-10,-20],
                        [-10,  0,  0,  0,  0,  0,  0,-10],
                        [-10,  0,  5, 10, 10,  5,  0,-10],
                        [-10,  5,  5, 10, 10,  5,  5,-10],
                        [-10,  0, 10, 10, 10, 10,  0,-10],
                        [-10, 10, 10, 10, 10, 10, 10,-10],
                        [-10,  5,  0,  0,  0,  0,  5,-10],
                        [-20,-10,-10,-10,-10,-10,-10,-20]
                    ],
                'R': [
                        [0,  0,  0,  0,  0,  0,  0,  0],
                        [5, 10, 10, 10, 10, 10, 10,  5],
                        [-5,  0,  0,  0,  0,  0,  0, -5],
                        [-5,  0,  0,  0,  0,  0,  0, -5],
                        [-5,  0,  0,  0,  0,  0,  0, -5],
                        [-5,  0,  0,  0,  0,  0,  0, -5],
                        [-5,  0,  0,  0,  0,  0,  0, -5],
                        [0,  0,  0,  5,  5,  0,  0,  0]
                    ],
                'Q':[
                        [-20,-10,-10, -5, -5,-10,-10,-20],
                        [-10,  0,  0,  0,  0,  0,  0,-10],
                        [-10,  0,  5,  5,  5,  5,  0,-10],
                        [-5,  0,  5,  5,  5,  5,  0, -5],
                        [0,  0,  5,  5,  5,  5,  0, -5],
                        [-10,  5,  5,  5,  5,  5,  0,-10],
                        [-10,  0,  5,  0,  0,  0,  0,-10],
                        [-20,-10,-10, -5, -5,-10,-10,-20]
                    ],
                'K':[
                        [-30,-40,-40,-50,-50,-40,-40,-30],
                        [-30,-40,-40,-50,-50,-40,-40,-30],
                        [-30,-40,-40,-50,-50,-40,-40,-30],
                        [-30,-40,-40,-50,-50,-40,-40,-30],
                        [-20,-30,-30,-40,-40,-30,-30,-20],
                        [-10,-20,-20,-20,-20,-20,-20,-10],
                        [-10,0,-10,-20,-20,-10,0,-10],
                        [20, 30, 10,  0,  0, 10, 30, 20]
                    ]
                }
KING_ENDGAME_SCORE = [  # King endgame table - encourages moving to the center
    [-50,-40,-30,-20,-20,-30,-40,-50],
    [-30,-20,-10,  0,  0,-10,-20,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 30, 40, 40, 30,-10,-30],
    [-30,-10, 20, 30, 30, 20,-10,-30],
    [-30,-30,  0,  0,  0,  0,-30,-30],
    [-50,-30,-30,-30,-30,-30,-30,-50]
]

TRANSPOSITION_TABLE = {}

# Flags for the table
FLAG_EXACT = 0
FLAG_LOWERBOUND = 1
FLAG_UPPERBOUND = 2


class MoveContextManager:
    def __init__(self, board: Board, move: Move):
        self.board = board
        self.move = move
        self.eaten = board.piece_at(*self.move.dst)
        self.piece = board.piece_at(*self.move.src)

    def __enter__(self):
        self.board.make(self.move)

    def __exit__(self, exc_type, exc_val, exc_tb):
        (sr, sc), (dr, dc), promote = self.move
        rev_move = Move((dr, dc), (sr, sc), None)
        self.board.make(rev_move)
        self.board.board[dr][dc] = self.eaten
        if promote:
            self.board.board[sr][sc] = self.piece


def get_board_key(board):
    """
    Hash function for bored to moves_cache
    :param board: the board to hash
    :return: the hash key
    """
    return board.turn, tuple(tuple(row) for row in board.board)


def store_tt(board: Board, depth: int, score: float, flag: int, best_move: Optional[Move]):
    """Stores a position in the Transposition Table."""
    key = get_board_key(board)
    TRANSPOSITION_TABLE[key] = {
        'depth': depth,
        'score': score,
        'flag': flag,
        'move': best_move
    }


def probe_tt(board: Board, depth: int, alpha: float, beta: float) -> Tuple[Optional[float], Optional[Move]]:
    """
    Checks if the current board position is already in the table.
    Returns (score, best_move) if usable, else (None, best_move_for_ordering).
    """
    key = get_board_key(board)
    entry = TRANSPOSITION_TABLE.get(key)

    if not entry:
        return None, None

    # We can always use the cached move for ordering, even if depth is low
    cached_move = entry['move']

    # We can only use the SCORE if the cached depth is sufficient
    if entry['depth'] >= depth:
        if entry['flag'] == FLAG_EXACT:
            return entry['score'], cached_move
        elif entry['flag'] == FLAG_LOWERBOUND:
            if entry['score'] >= beta:
                return entry['score'], cached_move
        elif entry['flag'] == FLAG_UPPERBOUND:
            if entry['score'] <= alpha:
                return entry['score'], cached_move

    return None, cached_move


def choose_random_move(board: Board) -> Move:
    """Return a uniformly random legal move or None if no moves exist."""
    legal = board.legal_moves()
    return random.choice(legal) if legal else None


def is_endgame(board: Board) -> bool:
    """Check if we are in the endgame phase (no queens usually indicates endgame)."""
    queens_count = 0
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(r, c)
            if piece and piece[1] == 'Q':
                queens_count += 1
    return queens_count == 0


def get_pawn_structure_score(board: Board) -> int:
    """Calculates penalties for doubled and isolated pawns."""
    white_score = 0
    black_score = 0
    white_pawns = [0] * 8
    black_pawns = [0] * 8

    # Map pawn counts per column
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(r, c)
            if piece == ('w', 'P'):
                white_pawns[c] += 1
            elif piece == ('b', 'P'):
                black_pawns[c] += 1

    # Calculate penalties
    for c in range(8):
        # Penalty for Doubled Pawns
        if white_pawns[c] > 1: white_score -= 20
        if black_pawns[c] > 1: black_score -= 20

        # Penalty for Isolated Pawns
        # For White:
        if white_pawns[c] > 0:
            has_left = (c > 0 and white_pawns[c - 1] > 0)
            has_right = (c < 7 and white_pawns[c + 1] > 0)
            if not has_left and not has_right:
                white_score -= 30

        # For Black:
        if black_pawns[c] > 0:
            has_left = (c > 0 and black_pawns[c - 1] > 0)
            has_right = (c < 7 and black_pawns[c + 1] > 0)
            if not has_left and not has_right:
                black_score -= 30

    return white_score - black_score


def get_loc_score(board: Board, r: int, c: int) -> int:
    """
    add location bonus to heuristic function based on piece location on board.
    :param board: current state of the board.
    :param r: row index.
    :param c: column index.
    :return: location bonus score
    """
    if board.piece_at(r, c):
        piece = board.piece_at(r, c)
        piece_sign = 1 if piece[0] == 'w' else -1
        return (PIECES_LOCATION_SCORE[piece[1]][r][c] + PIECES_SCORE[piece[1]]) * piece_sign
    return 0


def raw_heuristic(board: Board) -> int:
    score = 0
    endgame_phase = is_endgame(board)

    # 1. Material & Location Score
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(r, c)
            if piece:
                piece_color = piece[0]  # 'w' or 'b'
                piece_type = piece[1]  # 'P', 'N', 'B', ...
                sign = 1 if piece_color == 'w' else -1

                table_r = r if piece_color == 'w' else 7 - r

                # Choose scoring table: use endgame table for King if in endgame
                if piece_type == 'K' and endgame_phase:
                    location_bonus = KING_ENDGAME_SCORE[table_r][c]
                else:
                    location_bonus = PIECES_LOCATION_SCORE[piece_type][table_r][c]

                material_value = PIECES_SCORE[piece_type]
                score += (material_value + location_bonus) * sign

    # 2. Pawn Structure Score
    score += get_pawn_structure_score(board)

    return score


def quiescence_max(board: Board, alpha: float, beta: float) -> float:
    """
    Quiescence search for the maximizing player (White).
    Only searches capture moves to avoid the horizon effect.
    """
    stand_pat = raw_heuristic(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    for move in board.legal_moves():
        if board.piece_at(move.dst[0], move.dst[1]):
            with MoveContextManager(board, move):
                score = quiescence_min(board, alpha, beta)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
    return alpha


def quiescence_min(board: Board, alpha: float, beta: float) -> float:
    """
    Quiescence search for the minimizing player (Black).
    Only searches capture moves to avoid the horizon effect.
    """
    stand_pat = raw_heuristic(board)
    if stand_pat <= alpha:
        return alpha
    if stand_pat < beta:
        beta = stand_pat
    for move in board.legal_moves():
        if board.piece_at(move.dst[0], move.dst[1]):
            with MoveContextManager(board, move):
                score = quiescence_max(board, alpha, beta)
                if score <= alpha:
                    return alpha
                if score < beta:
                    beta = score
    return beta


def evaluate(board: Board) -> float:
    """
    Calculate heuristic score from White's perspective.
    :param board: current state of the board.
    :return: heuristic score
    """
    # check terminal state of game
    if is_terminal(board):
        if board.turn == 'white':
            return -float('inf')
        return float('inf')
    return raw_heuristic(board)


def is_terminal(board: Board) -> bool:
    """
    Return True if the board is terminal (game over).
    """
    res = board.outcome()
    if res is not None and res[0] == 'checkmate':
        return True
    return False


def get_move_score(board: Board, move: Move) -> int:
    """
    evaluate score of single move to determine order of going over moves in the alpha-bete pruning
    :param board: current state of the board.
    :param move: move to evaluate.
    :return: score of move
    """
    start_r, start_c = move.src
    end_r, end_c = move.dst

    target_piece = board.piece_at(end_r, end_c)
    score = 0

    if target_piece:
        attacker_piece = board.piece_at(start_r, start_c)
        victim_val = PIECES_SCORE.get(target_piece[1], 0)
        attacker_val = PIECES_SCORE.get(attacker_piece[1], 0) if attacker_piece else 0
        score = 10 * victim_val - attacker_val

    if move.promote:
        score += 1000

    return score


def minmax_max_component(board: Board, depth: int, nodes_visited: Set[Board]) -> Tuple[float, Optional[Move]]:
    """
    Goes over all possibilities to move and return the one that that will give us the biggest value according to minmax algorithm
    :param board: current state of the board.
    :param depth: depth to go into game branches
    :param nodes_visited: set of boards visited while calculating next move.
    :return: tuple of best move score (highest) and move itself
    """
    nodes_visited.add(board.clone())
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = -float('inf')
    for move in board.legal_moves():
        with MoveContextManager(board, move):
            value, made_move = minmax_min_component(board, depth - 1, nodes_visited)
            if value > best_score:
                best_score = value
                best_move = move
    return best_score, best_move


def minmax_min_component(board: Board, depth: int, nodes_visited: Set[Board]) -> Tuple[float,  Optional[Move]]:
    """
    Goes over all possibilities to move and return the one that that will give us the smallest value according to minmax algorithm
    :param board: current state of the board.
    :param depth: depth to go into game branches
    :param nodes_visited: set of boards visited while calculating next move.
    :return: tuple of best move score (lowest) and move itself
    """
    nodes_visited.add(board.clone())
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = float('inf')
    for move in board.legal_moves():
        with MoveContextManager(board, move):
            value, made_move = minmax_max_component(board, depth - 1, nodes_visited)
            if value < best_score:
                best_score = value
                best_move = move
    return best_score, best_move


def choose_minimax_move(board: Board, depth: int=2, metrics=None) -> Tuple[Move, Set[Board]]:
    """
    Pick a move for the current player using minimax (no pruning).

    Returns:
        (best_move, nodes_visited)
    """
    nodes_visited = set()
    if board.turn == 'white':
        best_score, best_move = minmax_max_component(board, depth, nodes_visited)
    else:
        best_score, best_move = minmax_min_component(board, depth, nodes_visited)
    return best_move, nodes_visited


def get_ordered_moves(board: Board, tt_move: Optional[Move] = None):
    """
    Order the legal moves properly to save time while branching
    """
    legal_moves = list(board.legal_moves())

    if not legal_moves:
        return []

    # If we have a move from the Transposition Table, try it first!
    if tt_move:
        # Move the tt_move to the front if it is in legal_moves
        for i, move in enumerate(legal_moves):
            if move.src == tt_move.src and move.dst == tt_move.dst:
                legal_moves.pop(i)
                legal_moves.insert(0, move)
                break

    # Sort the rest using MVV-LVA (Captures first)
    # We leave the first move alone if it was the TT move
    start_index = 1 if tt_move else 0
    legal_moves[start_index:] = sorted(legal_moves[start_index:], key=lambda m: get_move_score(board, m), reverse=True)

    return legal_moves


def alpha_beta_max_component(board: Board, depth: int, nodes_visited: Set[Board],
                             alpha: float = -float('inf'), beta: float = float('inf')) -> Tuple[float,  Optional[Move]]:
    """
    Goes over possibilities to move and return the one that that will give us the biggest value according to minmax algorithm.
    Uses alpha-beta pruning to avoid exploring dead cause branches and save compute time
    :param board: current state of the board.
    :param depth: depth to go into game branches
    :param nodes_visited: set of boards visited while calculating next move.
    :param alpha: lower bound on the score we already secured in previous branches
    :param beta: upper bound on the score we already secured in previous branches
    :return: tuple of best move score (highest) and move itself
    """

    tt_score, tt_move = probe_tt(board, depth, alpha, beta)
    if tt_score is not None:
        # If we got a valid score from cache, return it immediately!
        nodes_visited.add(board)
        return tt_score, tt_move

    nodes_visited.add(board)
    original_alpha = alpha

    if depth == 0:
        if not board.is_check(board.turn):
            return quiescence_max(board, alpha, beta), None
        if not list(board.legal_moves()):
            return float('-inf'), None
        return raw_heuristic(board), None

    if is_terminal(board):
        return evaluate(board), None

    best_move = None
    best_score = -float('inf')
    legal_moves = get_ordered_moves(board, tt_move)
    if not legal_moves:
        return evaluate(board), None
    for move in legal_moves:
        with MoveContextManager(board, move):
            value, _ = alpha_beta_min_component(board, depth - 1, nodes_visited, alpha, beta)
            if value > best_score:
                best_score = value
                best_move = move
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
    flag = FLAG_EXACT
    if best_score <= original_alpha:
        flag = FLAG_UPPERBOUND
    elif best_score >= beta:
        flag = FLAG_LOWERBOUND
    store_tt(board, depth, best_score, flag, best_move)
    return best_score, best_move


def alpha_beta_min_component(board: Board, depth: int, nodes_visited: Set[Board],
                             alpha: float = -float('inf'), beta: float = float('inf')) -> Tuple[float,  Optional[Move]]:
    """
    Goes over possibilities to move and return the one that that will give us the smallest value according to minmax algorithm.
    Uses alpha-beta pruning to avoid exploring dead cause branches and save compute time.
    :param board: current state of the board.
    :param depth: depth to go into game branches
    :param nodes_visited: set of boards visited while calculating next move.
    :param alpha: lower bound on the score we already secured in previous branches
    :param beta: upper bound on the score we already secured in previous branches
    :return: tuple of best move score (lowest) and move itself
    """
    tt_score, tt_move = probe_tt(board, depth, alpha, beta)
    if tt_score is not None:
        nodes_visited.add(board)
        return tt_score, tt_move

    nodes_visited.add(board)
    original_beta = beta  # Save for flag calculation
    if depth == 0:
        if not board.is_check(board.turn):
            return quiescence_min(board, alpha, beta), None
        if not list(board.legal_moves()):
            return float('inf'), None
        return raw_heuristic(board), None

    best_move = None
    best_score = float('inf')
    legal_moves = get_ordered_moves(board, tt_move)
    if not legal_moves:
        return evaluate(board), None
    for move in legal_moves:
        with MoveContextManager(board, move):
            value, made_move = alpha_beta_max_component(board, depth - 1, nodes_visited, alpha, beta)
            if value < best_score:
                best_score = value
                best_move = move
                beta = min(beta, best_score)
                if alpha >= beta:
                    break
    flag = FLAG_EXACT
    if best_score <= alpha:
        flag = FLAG_UPPERBOUND
    elif best_score >= original_beta:
        flag = FLAG_LOWERBOUND
    store_tt(board, depth, best_score, flag, best_move)
    return best_score, best_move


def choose_alphabeta_move(board: Board, depth: int=3, metrics=None):
    """
    Pick a move for the current player using minimax with alpha-beta pruning.

    Returns:
        (best_move, nodes_visited)
    """
    nodes_visited = set()
    if board.turn == 'white':
        best_score, best_move = alpha_beta_max_component(board, depth, nodes_visited)
    else:
        best_score, best_move = alpha_beta_min_component(board, depth, nodes_visited)
    return best_move, nodes_visited


def choose_move(board: Board):
    # iterative deeping
    for depth in range(1, 50):
        best_move, nodes_visited = choose_alphabeta_move(board, depth)
        if best_move:
            yield best_move

