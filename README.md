# Abstract
This project is a chess app as part of a group project for the course CSE150B in UCSD term fall 25. As an AI course, in this project the focus is on the AI agent that plays chess and not about other aspects of the app which was provided to us by the course staff. The AI algorithms used for the agent are minmax and alpha-beta pruning. After implementing the agent, I noticed that for depth greater than 2 the agent takes too much time to calculate the next move because of the branching factor in chess games. This is why in the future I believe I'll use more sophisticated ones to avoid pointless branching and to explore deeper more promising plays.

I chose to work solo on this project since the instructions were to implement only minmax and alpha-beta, where the only real thinking work is on the heuristic function


# Project Contains
Self-contained chess GUI + AI assignment.

- Rules: all standard moves **except** castling & en passant (omitted for clarity). Promotion->Queen.
- GUI: click piece then destination. Modes: Human vs Human, Human vs AI, AI vs AI.
- Supported AI algorithms: random, eval, minimax, alphabeta. All AI implementation code in located at `chesslab/ai/ai.py`


# AI Agent Algorithm Explanation
The base AI algorithm I rely on in this project is alpha-beta pruning. But I added some changes to the algorithm to improve performances:
  - To handel time limits, I added IDS that yields The best move we found in every layer (to handle time limits per play)
  - As deeper I can go the better the decision the agent is going to make. To go deeper, we need to work faster (time limit). 
One way is to make the pruning happen faster. Therefore, the boards to explore are sorted based on board evaluation so we estimate which 
  move will be the best in advance (asked Gemini: "How can I make the pruning happen faster?")
    - In addition, Since in chess You can get to the same board by different series of actions when we can remember boards that we got and
      write what is the best move that we calculated when we visited this board last time which is very likely to be still the 
      best move therefore we'll put the best move we calculated in the past in the top of the order. This technic called
      Transposition Table. (asked Gemini: "Is There any way to use boards I already visited in my advantaged to save time")

In addition, I tried to avoid as much as I can from Board.clone() since it's a bottleneck. To do so: 
- Wrote the heuristic function to minimize the calls to legal_moves since this function uses a lot of deep copies
- Added a context manager for branching so when I go deep I do the moves on the actual board game and when I return, 
I undo the move.


# Versions

  - MS1- Normal minmax and alpha-bete pruning 
  - MS2- Tournament Code. IDS based on alpha-beta pruning with minimum bottlenecks


**Since I don't play chess I Used AI to get the evaluation functions. (asked Gemini: "Can you give me heuristic function for chess?")