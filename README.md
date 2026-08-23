# Chess Engine

A fully-featured UCI-compatible chess engine of **~2500 Elo** ± 100 ELO estimated rating.

## core
- 120-sqaure mailbox (10x12)
- principal variation search with iterative deepening
- Alpha-Beta Pruning with aspiration windows
- Tapered Evaulation
- Move Ordering: TT > MVV-LVA > Queen promotion > killers > history
- Fully legal move validation

## Compile
```bash
g++ -O2 -Wall -Wextra -std=c++11 main.cpp -o chess_engine_uci.exe 2>&1
```

## Rules Supported
- [x] castling
- [x] en passant
- [x] pawn promotion
- [x] 3-fold repition draws
- [x] 50-move rule
- [x] insufficient material (KvK, K+NvK, K+BvK, K+BvK+B same color)

| Feature | Implementation |
|---------|----------------|
| **Search** | PVS + NMP (R=2/3) + LMR (depth≥3) + check extensions |
| **Quiescence** | Capture-only + SEE pruning + delta pruning |
| **Transposition Table** | 1M entries, Zobrist hashing, depth-preferred replacement |
| **Evaluation** | Material + PSTs (MG/EG) + bishop pair + pawn structure (passed/isolated/doubled) + rook files + king safety + development |
| **Opening Book** | 50 hardcoded lines (Flank/d4/e4) |
| **Time Management** | Clock-based allocation with increment, iterative deepening |
| **UCI Protocol** | Full: `uci`, `isready`, `ucinewgame`, `position`, `go`, `quit` |
| **Draw Detection** | 3-fold repetition, 50-move rule, insufficient material |

## Performance
 - **~2500 Elo** ± 100 ELO with fixed depth 6 (if no time or depth specified)
 - 200+ games vs stockfish (1350-2700 levels)
 - can go up to depth 64 within the allocated time

## limitations
 - Low nodes achieved at a specified depth
 - Overall accuracy ~90%
 - Reduced accuracy in opening and middlegames

## Sources
 - [Chess Programming Wiki](https://chessprogramming.org/) for concise documentation of different algorithms
 - [How To Write A Chess Programm in QBASIC By Dean Menezes](http://www.petesqbsite.com/sections/express/issue23/Tut_QB_Chess.txt) for giving ideas on how to write the code structure.
 - [datavorous/sameshi](https://github.com/datavorous/sameshi) for his readable `sameshi.h` file.