"""
Core AI routines for ChessLab.

All of the student work for this assignment lives in this file. Implement the
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


def choose_random_move(board: Board) -> Move:
    """Return a uniformly random legal move or None if no moves exist."""
    legal = board.legal_moves()
    return random.choice(legal) if legal else None


def evaluate(board: Board) -> float:
    """Return a heuristic score from White's perspective."""
    raise NotImplementedError("Implement your evaluation function in ai.py")


def is_terminal(board: Board) -> bool:
    res = board.outcome()
    if res is not None and res[0] == 'checkmate':
        return True
    return False


def minmax_max_component(board: Board, depth: int, nodes_visited: Set[Board]) -> Tuple[float, Optional[Move]]:
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = -float('inf')
    for move in board.legal_moves():
        new_board = board.clone()
        new_board.make(move)
        value, made_move = minmax_min_component(new_board, move, depth - 1, nodes_visited)
        if value > best_score:
            best_score = value
            best_move = move
    return best_score, best_move



def minmax_min_component(board: Board, depth: int, nodes_visited: Set[Board]) -> Tuple[float,  Optional[Move]]:
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = float('inf')
    for move in board.legal_moves():
        new_board = board.clone()
        new_board.make(move)
        value, made_move = minmax_min_component(new_board, move, depth - 1, nodes_visited)
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


def alpha_beta_max_component(board: Board, depth: int, nodes_visited: Set[Board],
                             alpha: float = -float('inf'), beta: float = float('inf')) -> Tuple[float,  Optional[Move]]:
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = -float('inf')
    for move in board.legal_moves():
        new_board = board.clone()
        new_board.make(move)
        value, made_move = alpha_beta_min_component(new_board, move, depth - 1, nodes_visited, alpha, beta)
        if value > best_score:
            best_score = value
            best_move = move
            alpha = max(alpha, best_score)
            if alpha >= beta:
                return best_score, best_move
    return best_score, best_move



def alpha_beta_min_component(board: Board, depth: int, nodes_visited: Set[Board],
                             alpha: float = -float('inf'), beta: float = float('inf')) -> Tuple[float,  Optional[Move]]:
    nodes_visited.add(board)
    if depth == 0 or is_terminal(board):
        return evaluate(board), None
    best_move = None
    best_score = float('inf')
    for move in board.legal_moves():
        new_board = board.clone()
        new_board.make(move)
        value, made_move = alpha_beta_min_component(new_board, move, depth - 1, nodes_visited, alpha, beta)
        if value < best_score:
            best_score = value
            best_move = move
            beta = min(beta, best_score)
            if alpha >= beta:
                return best_score, best_move
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
