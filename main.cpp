#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace chess
{

    constexpr int BOARD_SIZE = 120;
    constexpr int EMPTY = 0;
    constexpr int OFF_BOARD = 7;

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

    enum PieceType : int
    {
        PAWN = 1,
        KNIGHT = 2,
        BISHOP = 3,
        ROOK = 4,
        QUEEN = 5,
        KING = 6
    };

    enum Color : int
    {
        WHITE = 1,
        BLACK = -1
    };

    /*
    piece values:
    1: pawn
    3: knight
    3: bishop
    5: rook
    9: queen
    99: king
    */
    constexpr std::array<int, 7> PIECE_VALUES = {0, 1, 3, 3, 5, 9, 99};

    // Position Offsets
    constexpr std::array<int, 8> KNIGHT_OFFSETS = {-21, -19, -12, -8, 8, 12, 19, 21}; // Knight
    /*
    rooks use the offsets in 0-3rd pos, i.e. {-1, 1, -10, 10}
    bishops will use the offsets in 4-7th pos, i.e., {-11, -9, 9, 11}, queens will use all 8
    but the queen will require a looping algo to find moves
    */
    constexpr std::array<int, 8> KING_OFFSETS = {-1, 1, -10, 10, -11, -9, 9, 11};

    /*
    best_source: best square to move the piece from
    best_dest: best square to move the piece to
    board: Board representation, mailbox method
    */
    struct Position
    {
        std::array<int, BOARD_SIZE> board{};
        int best_source = 0;
        int best_dest = 0;

        // init board
        void init()
        {
            for (int i = 0; i < BOARD_SIZE; ++i)
            {
                int row = i / 10;
                int col = i % 10;
                int back_piece = "42356324"[col - 1] - '0';
                if (row < 2 || row > 9 || col < 1 || col > 8)
                {
                    board[i] = OFF_BOARD;
                }
                else if (row == 3)
                {
                    board[i] = PAWN;
                }
                else if (row == 8)
                {
                    board[i] = -PAWN;
                }
                else if (row == 2)
                {
                    board[i] = back_piece;
                }
                else if (row == 9)
                {
                    board[i] = -back_piece;
                }
                else
                {
                    board[i] = EMPTY;
                }
            }
        }
    };

    inline int abs_val(int x) { return x < 0 ? -x : x; }

int evaluate(Position &pos, int side, int depth_rem, int alpha, int beta,
             int from, int to, int piece, int captured, int *has_legal_move);

    bool is_in_check(const Position &pos, int side);

    // the heart of the engine, the search function
    // s: side to move
    // depth_rem: depth remaining
    // alpha: alpha
    // beta: beta
    int search(Position &pos, int side, int depth_rem, int alpha, int beta)
    {
        // we first eval leaf nodes
        bool at_leaf = (depth_rem == 0);
        int has_legal_move = 0;

        if (at_leaf)
        {
            int score = 0;
            for (int i = 21; i < 99; ++i)
            {
                if (pos.board[i] != OFF_BOARD)
                {
                    int piece = pos.board[i];
                    score += (piece > 0) ? PIECE_VALUES[piece] : -PIECE_VALUES[-piece];
                }
            }
            score *= side;
            if (score > alpha)
                alpha = score;
            if (alpha >= beta)
                return beta;
        }
        // we sum material values for all pieces,
        // if score is positive, white is at advantage, if negative, black is at advantage
        // we multiply by s to get the perspective of the side to move

        // we use a two pass move generation approach
        // first we analyse captures (always executed)
        // quiet moves (skipped at leaf nodes)

        // something to be noted: at leaf nodes, we only captures to avoid HORIZON EFFECT
        // take a look into what "quiescence search" is all about
        int max_pass = at_leaf ? 1 : 2;
        for (int pass = 0; pass < max_pass; ++pass)
        {
            for (int from = 21; from < 99; ++from)
            {
                int piece = pos.board[from];
                if (piece == OFF_BOARD || piece == EMPTY || (piece > 0) != (side > 0))
                    continue;

                int type = abs_val(piece);

                // we generate the pawn moves
                if (type == PAWN)
                {
                    int fwd = (side == WHITE) ? 10 : -10;

                    if (pass == 0)
                    {
                        // the captures go here
                        for (int dx = -1; dx <= 1; dx += 2)
                        {
                            int to = from + fwd + dx;
                            int captured = pos.board[to];
                            if (captured && captured != OFF_BOARD && (captured > 0) != (side > 0))
                            {
                                alpha = evaluate(pos, side, depth_rem, alpha, beta, from, to, piece, captured, &has_legal_move);
                                if (alpha >= beta)
                                    return beta;
                            }
                        }
                    }
                    else if (!pos.board[from + fwd])
                    {
                        // the quiet move goes here (only if the square in front is empty)
                        int to = from + fwd;
                        alpha = evaluate(pos, side, depth_rem, alpha, beta, from, to, piece, 0, &has_legal_move);
                        if (alpha >= beta)
                            return beta;

                        // two squares from starting position
                        bool at_start = (side == WHITE) ? (from < 40) : (from > 70);
                        if (at_start && !pos.board[from + 2 * fwd])
                        {
                            to = from + 2 * fwd;
                            alpha = evaluate(pos, side, depth_rem, alpha, beta, from, to, piece, 0, &has_legal_move);
                            if (alpha >= beta)
                                return beta;
                        }
                    }
                }
                else
                {
                    // we generate other piece' moves

                    // this is the direction array
                    const std::array<int, 8> *dirs = &KING_OFFSETS;
                    int start_dir = 0, end_dir = 8;

                    if (type == KNIGHT)
                    {
                        // knight
                        dirs = &KNIGHT_OFFSETS;
                        start_dir = 0;
                        end_dir = 8;
                    }
                    else if (type == ROOK)
                    {
                        // rook
                        start_dir = 0;
                        end_dir = 4;
                    }
                    else if (type == BISHOP)
                    {
                        // bishop
                        start_dir = 4;
                        end_dir = 8;
                    }
                    else
                    {
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
                    bool is_slider = (type != KNIGHT && type != KING);
                    for (int i = start_dir; i < end_dir; ++i)
                    {
                        int step = (*dirs)[i];
                        int to = from;

                        while (true)
                        {
                            to += step;
                            int target = pos.board[to];

                            if (target == OFF_BOARD)
                                break;
                            if (target && (target > 0) == (side > 0))
                                break;

                            if (pass == 0)
                            {
                                if (target)
                                {
                                    alpha = evaluate(pos, side, depth_rem, alpha, beta, from, to, piece, target, &has_legal_move);
                                    if (alpha >= beta)
                                        return beta;
                                    break;
                                }
                            }
                            else
                            {
                                if (!target)
                                {
                                    alpha = evaluate(pos, side, depth_rem, alpha, beta, from, to, piece, 0, &has_legal_move);
                                    if (alpha >= beta)
                                        return beta;
                                }
                                else
                                {
                                    break;
                                }
                            }

                            if (!is_slider)
                                break;
                        }
                    }
                }
            }
        }
        return alpha;
    }

    // we execute move and evaluate here
    int evaluate(Position &pos, int side, int depth_rem, int alpha, int beta,
                 int from, int to, int piece, int captured, int *has_legal_move)
    {
        // make the move
        pos.board[to] = piece;
        pos.board[from] = EMPTY;

        // would leave king on check? if yes, undo and skip
        if (is_in_check(pos, side))
        {
            pos.board[from] = piece;
            pos.board[to] = captured;
            return alpha;
        }

        // mark that we found a legal move
        *has_legal_move = 1;

        // recursively search the resulting position (negamax)
        int score = -search(pos, -side, depth_rem ? depth_rem - 1 : 0, -beta, -alpha);

        // unmake the move
        pos.board[from] = piece;
        pos.board[to] = captured;

        // update best move if this is better
        if (score > alpha)
        {
            alpha = score;
            if (depth_rem == 5)
            {
                pos.best_source = from;
                pos.best_dest = to;
            }
        }

        // beta cutoff
        return alpha >= beta ? beta : alpha;
    }

    // figuring out whether king is in check of side s
    bool is_in_check(const Position &pos, int side)
    {
        int king_sq = 0;
        int enemy = -side;

        // first, locate our king on the board
        // we scan the playable area (indices 21-98 in the 120-square mailbox)
        for (int i = 21; i < 99; ++i)
        {
            if (pos.board[i] == KING * side)
            {
                king_sq = i;
                break;
            }
        }
        // no king found? shouldn't happen in valid positions, but guard anyway
        if (!king_sq)
            return false;

        // pawn attacks - pawns attack diagonally forward
        // white pawns attack from rank+1 (indices +9, +11 in 10x12 board)
        // black pawns attack from rank-1 (indices -9, -11)
        if (side == WHITE)
        {
            if (pos.board[king_sq + 9] == -PAWN || pos.board[king_sq + 11] == -PAWN)
                return true;
        }
        else
        {
            if (pos.board[king_sq - 9] == PAWN || pos.board[king_sq - 11] == PAWN)
                return true;
        }

        // knight attacks - check all 8 L-shaped knight moves
        // knights jump, so no need to check for blocking pieces
        for (int i = 0; i < 8; ++i)
            if (pos.board[king_sq + KNIGHT_OFFSETS[i]] == KNIGHT * enemy)
                return true;

        // king attacks - enemy king adjacent to our king (illegal in chess, but check anyway)
        // KING_OFFSETS covers all 8 surrounding squares
        for (int i = 0; i < 8; ++i)
            if (pos.board[king_sq + KING_OFFSETS[i]] == KING * enemy)
                return true;

        // sliding piece attacks: rooks and queens along orthogonal lines (indices 0-3)
        // KING_OFFSETS[0-3] = {-1, 1, -10, 10} = left, right, down, up
        for (int i = 0; i < 4; ++i)
        {
            int t = king_sq;
            while (true)
            {
                t += KING_OFFSETS[i];
                if (pos.board[t] == OFF_BOARD)
                    break;                    // hit board edge
                if (!pos.board[t])
                    continue;                 // empty square, keep sliding
                int pt = abs_val(pos.board[t]);
                // found enemy rook or queen on this ray
                if ((pos.board[t] > 0) == (enemy > 0) && (pt == ROOK || pt == QUEEN))
                    return true;
                break;                        // blocked by any piece (friend or other enemy)
            }
        }

        // sliding piece attacks: bishops and queens along diagonal lines (indices 4-7)
        // KING_OFFSETS[4-7] = {-11, -9, 9, 11} = four diagonals
        for (int i = 4; i < 8; ++i)
        {
            int t = king_sq;
            while (true)
            {
                t += KING_OFFSETS[i];
                if (pos.board[t] == OFF_BOARD)
                    break;
                if (!pos.board[t])
                    continue;
                int pt = abs_val(pos.board[t]);
                // found enemy bishop or queen on this ray
                if ((pos.board[t] > 0) == (enemy > 0) && (pt == BISHOP || pt == QUEEN))
                    return true;
                break;
            }
        }
        return false;
    }

    char piece_char(int piece)
    {
        static const char *symbols = " PNBRQK";
        if (piece == OFF_BOARD)
            return '#';
        if (piece == EMPTY)
            return '.';
        char c = symbols[abs_val(piece)];
        return (piece > 0) ? c : (c + 32);
    }

    // convert algebraic square (file 'a'-'h', rank '1'-'8') to 120-square index
    // returns -1 if invalid
    int sq(char file, char rank)
    {
        if (file < 'a' || file > 'h' || rank < '1' || rank > '8')
            return -1;
        return (rank - '0' + 1) * 10 + (file - 'a' + 1);
    }

    // print board with coordinates
    void board(const Position &pos)
    {
        puts("");
        puts("  a b c d e f g h");
        for (int r = 9; r >= 2; --r)
        {
            printf("%d ", r - 1);
            for (int c = 1; c <= 8; ++c)
            {
                printf("%c ", piece_char(pos.board[r * 10 + c]));
            }
            printf("%d\n", r - 1);
        }
        puts("  a b c d e f g h");
    }

    // make a move on the board (no validation)
    void make_move(Position &pos, int from, int to)
    {
        pos.board[to] = pos.board[from];
        pos.board[from] = EMPTY;
    }

    // parse move in algebraic notation (e.g., "e2e4")
    // returns true if valid
    bool parse_move(const char *str, int *from, int *to)
    {
        if (!str)
            return false;
        // skip whitespace
        while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r')
            ++str;
        if (strlen(str) < 4)
            return false;
        char f = str[0], r1 = str[1], t = str[2], r2 = str[3];
        *from = sq(f, r1);
        *to = sq(t, r2);
        return *from != -1 && *to != -1;
    }

} // namespace chess

// AI move using the search function
int ai_move(chess::Position &pos, int side, int depth)
{
    int score = chess::search(pos, side, depth, -30000, 30000);
    return pos.best_source * 1000 + pos.best_dest;
}

// convert 120-square index to algebraic notation
void square_to_algebraic(int sq, char *buf)
{
    int file = (sq % 10) - 1;
    int rank = (sq / 10) - 1;
    buf[0] = 'a' + file;
    buf[1] = '1' + rank;
    buf[2] = '\0';
}

int main()
{
    chess::Position pos;
    pos.init();
    char input[16];
    int side = chess::WHITE;
    int human_side = chess::WHITE;
    bool vs_ai = false;

    puts("Play against AI? (y/n): ");
    if (fgets(input, sizeof(input), stdin) && (input[0] == 'y' || input[0] == 'Y'))
    {
        vs_ai = true;
        puts("Play as White (w) or Black (b)? ");
        if (fgets(input, sizeof(input), stdin) && (input[0] == 'b' || input[0] == 'B'))
            human_side = chess::BLACK;
    }

    while (true)
    {
        chess::board(pos);

        if (vs_ai && side != human_side)
        {
            // AI's turn
            int move = ai_move(pos, side, 5);
            int from = move / 1000;
            int to = move % 1000;
            char from_alg[3], to_alg[3];
            square_to_algebraic(from, from_alg);
            square_to_algebraic(to, to_alg);
            printf("AI plays: %s%s\n", from_alg, to_alg);
            chess::make_move(pos, from, to);
            side = -side;
            continue;
        }

        printf("%s to move (e.g., e2e4): ", side == chess::WHITE ? "White" : "Black");
        if (!fgets(input, sizeof(input), stdin))
            break;

        int from, to;
        if (!chess::parse_move(input, &from, &to))
        {
            puts("Invalid move format. Use e2e4");
            continue;
        }

        int piece = pos.board[from];
        if (piece == chess::EMPTY || piece == chess::OFF_BOARD || (piece > 0) != (side > 0))
        {
            puts("No piece of yours there");
            continue;
        }

        chess::make_move(pos, from, to);
        side = -side;
    }
    return 0;
}