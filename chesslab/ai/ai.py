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
from typing import Optional, Tuple, Set, Dict

from ..board import Board, Move
from ..common.profiling import Counter

MoveType = Tuple[Tuple[int, int], Tuple[int, int], Optional[str]]
PIECES_SCORE = {'P':100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
PIECES_LOCATION_SCORE = {'P':[
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
                        [20, 20,  0,  0,  0,  0, 20, 20],
                        [20, 30, 10,  0,  0, 10, 30, 20]
                    ]
                }


def get_board_key(board):
    """
    Hash function for bored to moves_cache
    :param board: the board to hash
    :return: the hash key
    """
    return board.turn, tuple(tuple(row) for row in board.board)


def choose_random_move(board: Board) -> Move:
    """Return a uniformly random legal move or None if no moves exist."""
    legal = board.legal_moves()
    return random.choice(legal) if legal else None


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


def evaluate(board: Board) -> float:
    """
    Calculate heuristic score from White's perspective.
    :param board: current state of the board.
    :return: heuristic score
    """

    # check terminal state of game
    if is_terminal(board):
        if board.turn == 'white':
            return float('inf')
        return -float('inf')

    score = 0
    #gives pieces score based on their location
    for r in range(8):
        for c in range(8):
            if board.piece_at(r, c):
                score += get_loc_score(board, r, c)
    return score



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
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = -float('inf')
    for move in board.legal_moves():
        new_board = board.clone()
        new_board.make(move)
        value, made_move = minmax_min_component(new_board, depth - 1, nodes_visited)
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
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = float('inf')
    for move in board.legal_moves():
        new_board = board.clone()
        new_board.make(move)
        value, made_move = minmax_max_component(new_board, depth - 1, nodes_visited)
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


def alpha_beta_max_component(board: Board, depth: int, best_move_cache: Dict, nodes_visited: Set[Board],
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

    board_key = get_board_key(board)
    hash_move = best_move_cache.get(board_key)
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = -float('inf')
    legal_moves = list(board.legal_moves())
    legal_moves.sort(key=lambda m: (float('inf') if m.src == hash_move.src and m.dst == hash_move.dst else get_move_score(board, m)) if hash_move
                                else get_move_score(board, m), reverse=True)
    for move in legal_moves:
        new_board = board.clone()
        new_board.make(move)
        value, made_move = alpha_beta_min_component(new_board, depth - 1, best_move_cache, nodes_visited, alpha, beta)
        if value > best_score:
            best_score = value
            best_move = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                best_move_cache[board_key] = best_move
                return best_score, best_move
    best_move_cache[board_key] = best_move
    return best_score, best_move



def alpha_beta_min_component(board: Board, depth: int, best_move_cache: Dict, nodes_visited: Set[Board],
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
    board_key = get_board_key(board)
    hash_move = best_move_cache.get(board_key)
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = float('inf')
    legal_moves = list(board.legal_moves())
    legal_moves.sort(key=lambda m: (float('inf') if m.src == hash_move.src and m.dst == hash_move.dst else get_move_score(board, m)) if hash_move
                                else get_move_score(board, m), reverse=True)
    for move in legal_moves:
        new_board = board.clone()
        new_board.make(move)
        value, made_move = alpha_beta_max_component(new_board, depth - 1, best_move_cache, nodes_visited, alpha, beta)
        if value < best_score:
            best_score = value
            best_move = move
            beta = min(beta, best_score)
            if alpha >= beta:
                best_move_cache[board_key] = best_move
                return best_score, best_move
    best_move_cache[board_key] = best_move
    return best_score, best_move


def choose_alphabeta_move(board: Board, depth: int=3, metrics=None):
    """
    Pick a move for the current player using minimax with alpha-beta pruning.

    Returns:
        (best_move, nodes_visited)
    """
    move_cache = {}
    nodes_visited = set()
    best_move = None
    # iterative deeping
    for current_d in range(1, depth + 1):
        if board.turn == 'white':
            best_score, best_move = alpha_beta_max_component(board, current_d, move_cache,nodes_visited)
        else:
            best_score, best_move = alpha_beta_min_component(board, current_d, move_cache,nodes_visited)
    return best_move, nodes_visited
