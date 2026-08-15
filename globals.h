// Header files needed
#include <stdio.h>
#include <stdlib.h>

// shorten syntax
#define j(x) ((x) < 0 ? -(x) : (x))
#define l(i, a, b) for (int i = (a); i < (b); i++) // looping syntax

// Board represetation

/*
piece coding:
0: empty cell
1: pawn
2: knight
3: bishop
4: rook
5: queen
6: king
7: off-board square

+1: white
-1: black
 */

// Weights for evaluation

/*
piece values:
1: pawn
3: knight
3: bishop
5: rook
9: queen
99: king

int board[8][8] = {
    {  5,  3,  3,  9, 99,  3,  3,  5 }, // Rank 8 (Black Major Pieces)
    {  1,  1,  1,  1,  1,  1,  1,  1 }, // Rank 7 (Black Pawns)
    {  0,  0,  0,  0,  0,  0,  0,  0 }, // Rank 6
    {  0,  0,  0,  0,  0,  0,  0,  0 }, // Rank 5
    {  0,  0,  0,  0,  0,  0,  0,  0 }, // Rank 4
    {  0,  0,  0,  0,  0,  0,  0,  0 }, // Rank 3
    {  1,  1,  1,  1,  1,  1,  1,  1 }, // Rank 2 (White Pawns)
    {  5,  3,  3,  9, 99,  3,  3,  5 }  // Rank 1 (White Major Pieces)
};

 */
const int v[] = {0, 1, 3, 3, 5, 9, 99};

// Position Offsets
int N[] = {-21, -19, -12, -8, 8, 12, 19, 21}; // Knight
/*
rooks use the offsets in 0-3rd pos, i.e. {-1, 1, -10, 10}
bishops will use the offsets in 4-7th pos, i.e., {-11, -9, 9, 11}, queens will use all 8
but the queen will require a looping algo to find moves
*/
int K[] = {-1, 1, -10, 10, -11, -9, 9, 11};

/*
best_source: best square to move the piece from
best_dest: best square to move the piece to
board: Board representation, mailbox method
*/
int best_source, best_dest, board[120];

// init board
void init()
{
    l(i, 0, 120)
    {
        int row = i / 10;
        int col = i % 10;
        int back_piece = "42356324"[col - 1] - '0';
        if (row < 2 || row > 9 || col < 1 || col > 8) {
            board[i] = 7;
        } else if (row == 3) {
            board[i] = 1;
        } else if (row == 8) {
            board[i] = -1;
        } else if (row == 2) {
            board[i] = back_piece;
        } else if (row == 9) {
            board[i] = -back_piece;
        } else {
            board[i] = 0;
        }
    }
}


// the heart of the engine, the search function
// s: side to move
// depth_rem: depth remaining
// alpha: alpha
// beta: beta
int search(int s, int depth_rem, int alpha, int beta) {
    // we first eval leaf nodes
    int at_leaf = !depth_rem;
    int has_legal_move = 0;

    if (at_leaf) {
        int score = 0;
        l(i, 21, 99) {
            if (board[i] != 7) score += (board[i] > 0) ? v[board[i]] : -v[-board[i]];
        }
        score *= s;
        if (score > alpha) alpha = score;
        if (alpha >= beta) return beta;
    }
    // we sum material values for all pieces, 
    // if score is positive, white is at advantage, if negative, black is at advantage
    // we multiply by s to get the perspective of the side to move

    // we use a two pass move generation approach 
    // first we analyse captures (always executed)
    // quiet moves (skipped at leaf nodes)

    // something to be noted: at leaf nodes, we only captures to avoid HORIZON EFFECT
    // take a look into what "quiescence search" is all about
    l(pass, 0, at_leaf ? 1 : 2) {
        l(from, 21, 99) {
            int piece = board[from];
            if (piece == 7 || !piece || (piece > 0) != (s > 0)) continue;

            int type = j(piece);

            // we generate the pawn moves
            if (type == 1) {
                int fwd = (s == 1) ? 10 : -10;

                if (!pass) {
                    // the captures go here
                    for (int dx = -1; dx <= 1; dx += 2) {
                        int to = from + fwd + dx;
                        int captured = board[to];
                        if (captured && captured != 7 && (captured > 0) != (s > 0)) {
                            alpha = E(s, depth_rem, alpha, beta, from, to, piece, captured, &has_legal_move);
                            if (alpha >= beta) return beta;
                        }
                    }
                } else if (!board[from + fwd]) {
                    // the quiet move goes here (only if the square in front is empty)
                    int to = from + fwd;
                    alpha = E(s, depth_rem, alpha, beta, from, to, piece, 0, &has_legal_move);
                    if (alpha >= beta) return beta;

                    // two squares from starting position
                    if (((s == 1 && from < 40) || (s == -1 && from > 70)) && !board[from + 2 * fwd]) {
                        to = from + 2 * fwd;
                        alpha = evaluate(s, depth_rem, alpha, beta, from, to, piece, 0, &has_legal_move);
                        if (alpha >= beta) return beta;
                    }
                }
            } else {
                // we generate other piece' moves

                // this is the direction array
                int *dirs = K;
                int start_dir, end_dir;

                if (type == 2) {
                    // knight 
                    dirs = N;
                    start_dir = 0;
                    end_dir = 8;
                } else if (type == 4) {
                    // rook
                    start_dir = 0;
                    end_dir = 4;
                } else if (type == 3) {
                    // bishop
                    start_dir = 4;
                    end_dir = 8;
                } else {
                    // queen
                    start_dir = 0;
                    end_dir = 8;
                }

                // what we did above is direction selection
                // knights: use the N[] array with all L shaped moves
                // rooks K[0-3]: othogonal directions
                // bishops K[4-7]: diagonal directions
                // kings/queens K[0-7]: all 8 directions

                // here we define the sliding logic for pieces
                // rooks, bishops and queens slide until blockde
                // non sliders move once per direction
                // captures and quiet moves are generated in the same loop, 
                // differentiated by the "pass" variable
                l(i, start_dir, end_dir) {
                    int step = dirs[i];
                    int to = from;
                    int is_slider = (type != 2 && type != 6);

                    while (1) {
                        to += step;
                        int target = board[to];

                        if (target == 7) break;
                        if (target && (target > 0) == (s > 0)) break;

                        if (!pass) {
                            if (target) {
                                alpha = E(s, depth_rem, alpha, beta, from, to, piece, target, &has_legal_move);
                                if (alpha >= beta) return beta;
                                break;
                            }
                        } else {
                            if (!target) {
                                alpha = evaluate(s, depth_rem, alpha, beta, from, to, piece, 0, &has_legal_move);
                                if (alpha >= beta) return beta;
                            } else {
                                break;
                            }
                        }

                        if (!is_slider) break;
                    }
                }
            }
        }
    }
    return alpha;
}

// figuring out whether king is in check of side s
int check(int s) {
    int king_sq = 0;
    int enemy = -s;

    l(i, 21, 99) {
        if (board[i] == 6 * s) {
            king_sq = i;
            break;
        }
    }
    if (!king_sq) return 0;

    if (s == 1) {
        if (board[king_sq + 9] == -1 || board[king_sq + 11] == -1) return 1;
    } else {
        if (board[king_sq - 9] == 1 || board[king_sq - 11] == 1) return 1;
    }

    l(i, 0, 8) if (board[king_sq + N[i]] == 2 * enemy) return 1;

    l(i, 0, 8) if (board[king_sq + K[i]] == 6 * enemy) return 1;

    l(i, 0, 4) {
        int t = king_sq;
        while (1) {
            t += K[i];
            if (board[t] == 7) break;
            if (!board[t]) continue;
            if ((board[t] > 0) == (enemy > 0) && (j(board[t]) == 4 || j(board[t]) == 5)) 
                return 1;
            break;
        }
    }

    l(i, 4, 8) {
        int t = king_sq;
        while (1) {
            t += K[i];
            if (board[t] == 7) break;
            if (!board[t]) continue;
            if ((board[t] > 0) == (enemy > 0) && (j(board[t]) == 3 || j(board[t]) == 5)) return 1;
            break;
        }
    }
    return 0;
}