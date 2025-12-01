# Abstract
This project is a chess app as part of a group project for the course CSE150B in UCSD term fall 25. As an AI course, in this project the focus is on the AI agent that plays chess and not about other aspects of the app which was provided to us by the course staff. The AI algorithms used for the agent are minmax and alpha-beta pruning. After implementing the agent, I noticed that for depth greater than 2 the agent takes too much time to calculate the next move because of the branching factor in chess games. This is why in the future I believe I'll use more sophisticated ones to avoid pointless branching and to explore deeper more promising plays.

I chose to work solo on this project since the instructions were to implement only minmax and alpha-beta, where the only real thinking work is on the heuristic function


# Project Contains
Self-contained chess GUI + AI assignment.

- Rules: all standard moves **except** castling & en passant (omitted for clarity). Promotion->Queen.
- GUI: click piece then destination. Modes: Human vs Human, Human vs AI, AI vs AI.
- Supported AI algorithms: random, eval, minimax, alphabeta. All AI implementation code in located at `chesslab/ai/ai.py`
