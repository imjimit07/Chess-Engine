#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
    struct Position;
    void compute_hash(Position &pos);

    struct Position
    {
        std::array<int, BOARD_SIZE> board{};
        int best_source = 0;
        int best_dest = 0;
        int best_promo = 0;
        uint8_t castling = 0b1111; // Bit 0: WK, Bit 1: WQ, Bit 2: BK, Bit 3: BQ
        int ep_sq = 0;             // en passant target square (0 = none)
        uint64_t hash = 0;         // Zobrist hash (pieces + castling + ep; no side bit)

        // init board
        void init()
        {
            castling = 0b1111;
            ep_sq = 0;
            best_source = 0;
            best_dest = 0;
            best_promo = 0;
            for (int i = 0; i < BOARD_SIZE; ++i)
            {
                int row = i / 10;
                int col = i % 10;
                if (row < 2 || row > 9 || col < 1 || col > 8)
                {
                    board[i] = OFF_BOARD;
                }
                else
                {
                    int back_piece = "42356324"[col - 1] - '0'; // now safe
                    if (row == 3)
                        board[i] = PAWN;
                    else if (row == 8)
                        board[i] = -PAWN;
                    else if (row == 2)
                        board[i] = back_piece;
                    else if (row == 9)
                        board[i] = -back_piece;
                    else
                        board[i] = EMPTY;
                }
            }
            compute_hash(*this);
        }
    };

    inline int abs_val(int x) { return x < 0 ? -x : x; }

    // Time management state shared by the search (set by ai_move).
    bool g_timed = false;
    bool g_timeout = false;
    long long g_nodes = 0;
    std::chrono::steady_clock::time_point g_deadline;

    // Piece-Square Tables, indexed by row (2..9 = rank 1..8) and col (1..8),
    // oriented for White. Black pieces use the vertically mirrored row (11 - row).
    constexpr std::array<int, 64> PAWN_PST = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0};
    constexpr std::array<int, 64> KNIGHT_PST = {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50};
    constexpr std::array<int, 64> BISHOP_PST = {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20};
    constexpr std::array<int, 64> ROOK_PST = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0};
    constexpr std::array<int, 64> QUEEN_PST = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20};
    constexpr std::array<int, 64> KING_MG_PST = {
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20};
    constexpr std::array<int, 64> KING_EG_PST = {
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50};

    // ---------------------------------------------------------------------
    // Zobrist hashing + transposition table
    // ---------------------------------------------------------------------
    // A move in the search. `captured` and `piece` are stored so move ordering
    // and make/unmake don't need to re-read the board.
    struct Move
    {
        int from, to, promo, captured, piece;
        int score;
    };

    const int MATE = 30000;
    const int MATE_THRESHOLD = 29000;
    enum TTFlag : uint8_t
    {
        TT_EXACT = 0,
        TT_LOWER = 1,
        TT_UPPER = 2
    };
    struct TTEntry
    {
        uint64_t key;
        int score;
        int depth;
        uint8_t flag;
        int from, to, promo;
    };
    std::vector<TTEntry> g_tt;
    int g_tt_mask;

    uint64_t g_zpiece[7][BOARD_SIZE]; // piece type (1-6) x square index
    uint64_t g_zcastling[16];         // castling rights mask (0-15)
    uint64_t g_zep[9];                // en passant file (0 = none)
    uint64_t g_zside;                 // side to move

    void tt_clear();

    uint64_t splitmix64(uint64_t &x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    void init_zobrist()
    {
        uint64_t s = 0x9E3779B97F4A7C15ULL;
        for (int t = 0; t < 7; ++t)
            for (int sq = 0; sq < BOARD_SIZE; ++sq)
                g_zpiece[t][sq] = splitmix64(s);
        for (int i = 0; i < 16; ++i)
            g_zcastling[i] = splitmix64(s);
        for (int i = 0; i < 9; ++i)
            g_zep[i] = splitmix64(s);
        g_zside = splitmix64(s);

        g_tt.assign(1 << 20, TTEntry());
        g_tt_mask = (1 << 20) - 1;
        tt_clear();
    }

    int ep_file(int sq) { return sq ? sq % 10 : 0; }

    // Full recompute of the hash from the board (used at setup).
    void compute_hash(Position &pos)
    {
        uint64_t h = 0;
        for (int i = 21; i < 99; ++i)
        {
            int p = pos.board[i];
            if (p == EMPTY || p == OFF_BOARD)
                continue;
            h ^= g_zpiece[abs_val(p)][i];
        }
        h ^= g_zcastling[pos.castling];
        h ^= g_zep[ep_file(pos.ep_sq)];
        pos.hash = h;
    }

    void tt_clear()
    {
        for (std::size_t i = 0; i < g_tt.size(); ++i)
            g_tt[i].key = 0;
    }

    void tt_store(uint64_t key, int depth, int score, int flag, int from, int to, int promo, int ply)
    {
        if (score >= MATE_THRESHOLD)
            score += ply;
        else if (score <= -MATE_THRESHOLD)
            score -= ply;
        TTEntry &e = g_tt[key & g_tt_mask];
        e.key = key;
        e.score = score;
        e.depth = depth;
        e.flag = flag;
        e.from = from;
        e.to = to;
        e.promo = promo;
    }

    int pst_value(int type, int row, int col, bool endgame)
    {
        int idx = (row - 2) * 8 + (col - 1);
        switch (type)
        {
        case PAWN:
            return PAWN_PST[idx];
        case KNIGHT:
            return KNIGHT_PST[idx];
        case BISHOP:
            return BISHOP_PST[idx];
        case ROOK:
            return ROOK_PST[idx];
        case QUEEN:
            return QUEEN_PST[idx];
        default:
            return endgame ? KING_EG_PST[idx] : KING_MG_PST[idx];
        }
    }

    // Static evaluation in centipawns from the perspective of `side`.
    // Material + piece-square tables + pawn structure + king shield.
    int evaluate_position(const Position &pos, int side)
    {
        int score = 0;
        int white_pawn_files[8] = {0};
        int black_pawn_files[8] = {0};
        int white_pawn_mask[8] = {0};
        int black_pawn_mask[8] = {0};
        int white_king = 0, black_king = 0;
        bool endgame = true;

        for (int i = 21; i < 99; ++i)
        {
            int piece = pos.board[i];
            if (piece == EMPTY || piece == OFF_BOARD)
                continue;
            int type = abs_val(piece);
            int row = i / 10, col = i % 10;
            if (type == QUEEN)
                endgame = false;

            if (piece > 0)
            {
                score += PIECE_VALUES[type] * 100;
                if (type == PAWN)
                {
                    ++white_pawn_files[col - 1];
                    white_pawn_mask[col - 1] |= 1 << (row - 2);
                    score += pst_value(PAWN, row, col, endgame);
                }
                else if (type == KING)
                    white_king = i;
                else
                    score += pst_value(type, row, col, endgame);
            }
            else
            {
                score -= PIECE_VALUES[type] * 100;
                if (type == PAWN)
                {
                    ++black_pawn_files[col - 1];
                    black_pawn_mask[col - 1] |= 1 << (row - 2);
                    score -= pst_value(PAWN, 11 - row, col, endgame);
                }
                else if (type == KING)
                    black_king = i;
                else
                    score -= pst_value(type, 11 - row, col, endgame);
            }
        }

        if (white_king)
            score += pst_value(KING, white_king / 10, white_king % 10, endgame);
        if (black_king)
            score -= pst_value(KING, 11 - black_king / 10, black_king % 10, endgame);

        for (int i = 21; i < 99; ++i)
        {
            int piece = pos.board[i];
            if (piece == EMPTY || piece == OFF_BOARD || abs_val(piece) != PAWN)
                continue;
            int row = i / 10, col = i % 10;
            int f = col - 1;

            if (piece > 0)
            {
                int adj = 0;
                for (int j = f - 1; j <= f + 1; ++j)
                    if (j >= 0 && j <= 7)
                        adj |= black_pawn_mask[j];
                bool passed = ((adj & (0xFF << (row - 1))) == 0);
                if (passed)
                    score += 10 + 15 * (row - 2);
                if (white_pawn_files[f] > 1)
                    score -= 10 * (white_pawn_files[f] - 1);
                bool isolated = (f > 0 ? white_pawn_files[f - 1] : 0) == 0 &&
                                (f < 7 ? white_pawn_files[f + 1] : 0) == 0;
                if (isolated)
                    score -= 10;
            }
            else
            {
                int adj = 0;
                for (int j = f - 1; j <= f + 1; ++j)
                    if (j >= 0 && j <= 7)
                        adj |= white_pawn_mask[j];
                bool passed = ((adj & ((1 << (row - 2)) - 1)) == 0);
                if (passed)
                    score -= 10 + 15 * (9 - row);
                if (black_pawn_files[f] > 1)
                    score += 10 * (black_pawn_files[f] - 1);
                bool isolated = (f > 0 ? black_pawn_files[f - 1] : 0) == 0 &&
                                (f < 7 ? black_pawn_files[f + 1] : 0) == 0;
                if (isolated)
                    score += 10;
            }
        }

        // King shield penalty during the middlegame (queens still on board):
        // penalize missing friendly pawns on the three files in front of the king.
        if (!endgame)
        {
            if (white_king && white_king / 10 <= 3)
            {
                int krow = white_king / 10, kcol = white_king % 10;
                int missing = 0;
                for (int cf = kcol - 1; cf <= kcol + 1; ++cf)
                {
                    if (cf < 1 || cf > 8)
                        continue;
                    for (int cr = krow + 1; cr <= krow + 2 && cr <= 9; ++cr)
                        if (pos.board[cr * 10 + cf] != PAWN)
                            ++missing;
                }
                score -= 15 * missing;
            }
            if (black_king && black_king / 10 >= 8)
            {
                int krow = black_king / 10, kcol = black_king % 10;
                int missing = 0;
                for (int cf = kcol - 1; cf <= kcol + 1; ++cf)
                {
                    if (cf < 1 || cf > 8)
                        continue;
                    for (int cr = krow - 1; cr >= krow - 2 && cr >= 2; --cr)
                        if (pos.board[cr * 10 + cf] != -PAWN)
                            ++missing;
                }
                score += 15 * missing;
            }
        }

        return score * side;
    }

    bool is_in_check(const Position &pos, int side);
    bool has_non_pawn_material(const Position &pos, int side);
    void toggle_move_hash(Position &pos, int from, int to, int piece, int captured, int promo,
                          int ep_removed, uint8_t old_castling, int old_ep);
    bool apply_move(Position &pos, int side, int from, int to, int piece, int captured, int promo,
                    int &ep_removed, uint8_t &old_castling, int &old_ep);
    void undo_move(Position &pos, int side, int from, int to, int piece, int captured, int promo,
                   int ep_removed, uint8_t old_castling, int old_ep);
    void add_move(Move *moves, int &n, int from, int to, int promo, int captured, int piece);
    void generate_moves(const Position &pos, int side, Move *moves, int &n, bool captures_only);
    void order_moves(Move *moves, int n, int tt_from, int tt_to, int tt_promo);
    int search(Position &pos, int side, int depth_rem, int alpha, int beta, int ply);

    // the heart of the engine, the search function
    // side: side to move
    // depth_rem: depth remaining
    // alpha/beta: search window
    // ply: plies from root (for mate scoring)
    int search(Position &pos, int side, int depth_rem, int alpha, int beta, int ply)
    {
        // periodic time check so the engine can never stall
        if (g_timeout || (++g_nodes % 1024 == 0 && g_timed && std::chrono::steady_clock::now() >= g_deadline))
        {
            g_timeout = true;
            return alpha;
        }

        bool at_leaf = (depth_rem == 0);
        uint64_t key = pos.hash ^ (side == WHITE ? 0 : g_zside);

        // transposition table probe (score cutoffs are never applied at the
        // root: ai_move needs an actual best move recorded)
        int tt_from = 0, tt_to = 0, tt_promo = 0;
        const TTEntry *e = &g_tt[key & g_tt_mask];
        if (e->key == key)
        {
            tt_from = e->from;
            tt_to = e->to;
            tt_promo = e->promo;
            if (ply > 0 && e->depth >= depth_rem)
            {
                int sc = e->score;
                if (sc >= MATE_THRESHOLD)
                    sc -= ply;
                else if (sc <= -MATE_THRESHOLD)
                    sc += ply;
                if (e->flag == TT_EXACT)
                    return sc;
                if (e->flag == TT_LOWER && sc >= beta)
                    return sc;
                if (e->flag == TT_UPPER && sc <= alpha)
                    return sc;
            }
        }

        // leaf: static eval (stand-pat), then captures only. If the side to
        // move is in check, extend one ply so escapes are searched: otherwise
        // mates at the horizon would be hidden behind a static evaluation.
        if (at_leaf && is_in_check(pos, side))
        {
            depth_rem = 1;
            at_leaf = false;
        }
        if (at_leaf)
        {
            int score = evaluate_position(pos, side);
            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }

        // null move pruning: if even giving up the move can't beat beta,
        // the real position is hopeless for the side to move.
        if (!at_leaf && depth_rem >= 3 && ply > 0 && !is_in_check(pos, side) &&
            has_non_pawn_material(pos, side) && has_non_pawn_material(pos, -side))
        {
            int saved_ep = pos.ep_sq;
            pos.hash ^= g_zep[ep_file(saved_ep)] ^ g_zep[0];
            pos.ep_sq = 0; // clear EP square during null move

            int null_score = -search(pos, -side, depth_rem - 3, -beta, -beta + 1, ply + 1);

            pos.ep_sq = saved_ep;
            pos.hash ^= g_zep[ep_file(saved_ep)] ^ g_zep[0];

            if (null_score >= beta)
                return beta;
        }

        // generate pseudo-legal moves (captures only at leaf nodes to avoid
        // the horizon effect) and order them: TT hash move first, then
        // captures by MVV-LVA, then quiet moves.
        Move moves[256];
        int n = 0;
        generate_moves(pos, side, moves, n, at_leaf);
        if (n == 0)
        {
            if (!at_leaf)
            {
                if (is_in_check(pos, side))
                    return -MATE + ply; // checkmate
                return 0;               // stalemate
            }
            return alpha;
        }
        order_moves(moves, n, tt_from, tt_to, tt_promo);

        int orig_alpha = alpha;
        bool has_legal = false;
        int best_from = 0, best_to = 0, best_promo = 0;

        for (int i = 0; i < n; ++i)
        {
            int from = moves[i].from, to = moves[i].to, promo = moves[i].promo;
            int captured = moves[i].captured, piece = moves[i].piece;

            int ep_removed = 0;
            uint8_t old_castling = pos.castling;
            int old_ep = pos.ep_sq;

            if (!apply_move(pos, side, from, to, piece, captured, promo, ep_removed, old_castling, old_ep))
                continue;

            has_legal = true;

            int score = -search(pos, -side, depth_rem ? depth_rem - 1 : 0, -beta, -alpha, ply + 1);

            undo_move(pos, side, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);

            if (score >= beta)
            {
                tt_store(key, depth_rem, score, TT_LOWER, from, to, promo, ply);
                return beta;
            }
            if (score > alpha)
            {
                alpha = score;
                best_from = from;
                best_to = to;
                best_promo = promo;
                if (ply == 0)
                {
                    pos.best_source = from;
                    pos.best_dest = to;
                    pos.best_promo = promo;
                }
            }
        }

        if (!has_legal)
        {
            if (!at_leaf)
            {
                if (is_in_check(pos, side))
                    return -MATE + ply; // checkmate
                return 0;               // stalemate
            }
        }

        if (has_legal)
        {
            if (alpha > orig_alpha)
                tt_store(key, depth_rem, alpha, TT_EXACT, best_from, best_to, best_promo, ply);
            else
                tt_store(key, depth_rem, alpha, TT_UPPER, 0, 0, 0, ply);
        }

        return alpha;
    }

    // toggle the Zobrist hash for a move being made or unmade (XOR is symmetric).
    void toggle_move_hash(Position &pos, int from, int to, int piece, int captured, int promo,
                          int ep_removed, uint8_t old_castling, int old_ep)
    {
        int type = abs_val(piece);
        pos.hash ^= g_zpiece[type][from];
        if (captured && !ep_removed)
            pos.hash ^= g_zpiece[abs_val(captured)][to];
        pos.hash ^= g_zpiece[promo ? promo : type][to];
        if (ep_removed)
            pos.hash ^= g_zpiece[PAWN][ep_removed];
        if (type == KING && abs(to - from) == 2)
        {
            int rfrom = (to == 27) ? 28 : (to == 23) ? 21
                                      : (to == 97)   ? 98
                                                     : 91;
            int rto = (to == 27) ? 26 : (to == 23) ? 24
                                    : (to == 97)   ? 96
                                                   : 94;
            pos.hash ^= g_zpiece[ROOK][rfrom] ^ g_zpiece[ROOK][rto];
        }
        if (pos.castling != old_castling)
            pos.hash ^= g_zcastling[old_castling] ^ g_zcastling[pos.castling];
        pos.hash ^= g_zep[ep_file(old_ep)] ^ g_zep[ep_file(pos.ep_sq)];
    }

    // Apply a pseudo-legal move. Returns true if the move leaves the king safe
    // (board left in the moved state; caller must call undo_move). Returns false
    // if illegal, with the board fully restored.
    bool apply_move(Position &pos, int side, int from, int to, int piece, int captured, int promo,
                    int &ep_removed, uint8_t &old_castling, int &old_ep)
    {
        old_castling = pos.castling;
        old_ep = pos.ep_sq;
        int fwd = (side == WHITE) ? 10 : -10;
        bool castle_move = (abs_val(piece) == KING && abs(to - from) == 2);
        ep_removed = 0;

        // cannot castle out of check
        if (castle_move && is_in_check(pos, side))
            return false;

        // castling transit check: the king passes through the mid square, which
        // must not be attacked. Checked while the rook still sits on its origin
        // square, so all ray attacks stay intact.
        if (castle_move)
        {
            int mid = (to == 27 || to == 97) ? to - 1 : to + 1;
            pos.board[from] = EMPTY;
            pos.board[mid] = piece;
            bool mid_safe = !is_in_check(pos, side);
            pos.board[mid] = EMPTY;
            pos.board[from] = piece;
            if (!mid_safe)
                return false;
        }

        // make the move
        pos.board[to] = promo ? (side > 0 ? promo : -promo) : piece;
        pos.board[from] = EMPTY;

        pos.ep_sq = 0;
        if (abs_val(piece) == PAWN && to == old_ep && (to - from == fwd + 1 || to - from == fwd - 1))
        {
            ep_removed = to - fwd;
            pos.board[ep_removed] = EMPTY;
        }
        else if (abs_val(piece) == PAWN && abs(to - from) == 20)
        {
            pos.ep_sq = from + fwd;
        }

        // castling: move rook along with the king
        if (castle_move)
        {
            if (to == 27)
            {
                pos.board[26] = pos.board[28];
                pos.board[28] = EMPTY;
            } // White O-O
            if (to == 23)
            {
                pos.board[24] = pos.board[21];
                pos.board[21] = EMPTY;
            } // White O-O-O
            if (to == 97)
            {
                pos.board[96] = pos.board[98];
                pos.board[98] = EMPTY;
            } // Black O-O
            if (to == 93)
            {
                pos.board[94] = pos.board[91];
                pos.board[91] = EMPTY;
            } // Black O-O-O
        }

        // update castling rights bitmask
        if (from == 25 || to == 25)
            pos.castling &= ~(1 | 2);
        if (from == 95 || to == 95)
            pos.castling &= ~(4 | 8);
        if (from == 28 || to == 28)
            pos.castling &= ~1;
        if (from == 21 || to == 21)
            pos.castling &= ~2;
        if (from == 98 || to == 98)
            pos.castling &= ~4;
        if (from == 91 || to == 91)
            pos.castling &= ~8;

        toggle_move_hash(pos, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);

        // would leave king on check? if yes, undo and skip
        if (is_in_check(pos, side))
        {
            undo_move(pos, side, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);
            return false;
        }
        return true;
    }

    // Unmake a move previously applied by apply_move.
    void undo_move(Position &pos, int side, int from, int to, int piece, int captured, int promo,
                   int ep_removed, uint8_t old_castling, int old_ep)
    {
        // toggle the hash back first (pos.castling/ep_sq are still the moved values)
        toggle_move_hash(pos, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);

        bool castle_move = (abs_val(piece) == KING && abs(to - from) == 2);
        pos.board[from] = piece;
        pos.board[to] = ep_removed ? EMPTY : captured;
        if (ep_removed)
            pos.board[ep_removed] = -PAWN * side;
        if (castle_move)
        {
            if (to == 27)
            {
                pos.board[28] = pos.board[26];
                pos.board[26] = EMPTY;
            }
            if (to == 23)
            {
                pos.board[21] = pos.board[24];
                pos.board[24] = EMPTY;
            }
            if (to == 97)
            {
                pos.board[98] = pos.board[96];
                pos.board[96] = EMPTY;
            }
            if (to == 93)
            {
                pos.board[91] = pos.board[94];
                pos.board[94] = EMPTY;
            }
        }
        pos.castling = old_castling;
        pos.ep_sq = old_ep;
    }

    void add_move(Move *moves, int &n, int from, int to, int promo, int captured, int piece)
    {
        if (n >= 256)
            return;
        moves[n].from = from;
        moves[n].to = to;
        moves[n].promo = promo;
        moves[n].captured = captured;
        moves[n].piece = piece;
        moves[n].score = 0;
        ++n;
    }

    // Generate pseudo-legal moves for `side`. If captures_only, only capturing
    // moves are produced (used at leaf nodes to avoid the horizon effect).
    void generate_moves(const Position &pos, int side, Move *moves, int &n, bool captures_only)
    {
        int fwd = (side == WHITE) ? 10 : -10;
        for (int from = 21; from < 99; ++from)
        {
            int piece = pos.board[from];
            if (piece == OFF_BOARD || piece == EMPTY || (piece > 0) != (side > 0))
                continue;

            int type = abs_val(piece);

            if (type == PAWN)
            {
                int promo_rank = (side == WHITE) ? 9 : 2;

                // captures
                for (int dx = -1; dx <= 1; dx += 2)
                {
                    int to = from + fwd + dx;
                    int captured = pos.board[to];
                    bool ep = (to == pos.ep_sq) && !captured;
                    if (ep)
                        captured = -PAWN * side;
                    if (ep || (captured && captured != OFF_BOARD && (captured > 0) != (side > 0)))
                    {
                        if (to / 10 == promo_rank)
                        {
                            for (int promo : {QUEEN, ROOK, BISHOP, KNIGHT})
                                add_move(moves, n, from, to, promo, captured, piece);
                        }
                        else
                        {
                            add_move(moves, n, from, to, 0, captured, piece);
                        }
                    }
                }

                // quiet pushes
                int to = from + fwd;
                bool is_promotion_push = (to / 10 == promo_rank);
                if ((!captures_only || is_promotion_push) && !pos.board[to])
                {
                    if (is_promotion_push)
                    {
                        for (int promo : {QUEEN, ROOK, BISHOP, KNIGHT})
                            add_move(moves, n, from, to, promo, 0, piece);
                    }
                    else
                    {
                        add_move(moves, n, from, to, 0, 0, piece);
                    }

                    bool at_start = (side == WHITE) ? (from < 40) : (from > 80);
                    if (!captures_only && at_start && !pos.board[from + 2 * fwd])
                        add_move(moves, n, from, from + 2 * fwd, 0, 0, piece);
                }
            }
            else
            {
                const std::array<int, 8> *dirs = &KING_OFFSETS;
                int start_dir = 0, end_dir = 8;
                if (type == KNIGHT)
                {
                    dirs = &KNIGHT_OFFSETS;
                }
                else if (type == ROOK)
                {
                    start_dir = 0;
                    end_dir = 4;
                }
                else if (type == BISHOP)
                {
                    start_dir = 4;
                    end_dir = 8;
                }

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
                        if (target)
                        {
                            add_move(moves, n, from, to, 0, target, piece);
                            break;
                        }
                        if (!captures_only)
                            add_move(moves, n, from, to, 0, 0, piece);
                        if (!is_slider)
                            break;
                    }
                }

                // castling moves (quiet only)
                if (type == KING && !captures_only)
                {
                    if (side == WHITE && from == 25)
                    {
                        if ((pos.castling & 1) && !pos.board[26] && !pos.board[27])
                            add_move(moves, n, 25, 27, 0, 0, piece);
                        if ((pos.castling & 2) && !pos.board[24] && !pos.board[23] && !pos.board[22])
                            add_move(moves, n, 25, 23, 0, 0, piece);
                    }
                    else if (side == BLACK && from == 95)
                    {
                        if ((pos.castling & 4) && !pos.board[96] && !pos.board[97])
                            add_move(moves, n, 95, 97, 0, 0, piece);
                        if ((pos.castling & 8) && !pos.board[94] && !pos.board[93] && !pos.board[92])
                            add_move(moves, n, 95, 93, 0, 0, piece);
                    }
                }
            }
        }
    }

    // Score each move and sort descending: TT hash move, then captures by
    // MVV-LVA (most valuable victim, least valuable attacker), then promotions,
    // then quiet moves.
    void order_moves(Move *moves, int n, int tt_from, int tt_to, int tt_promo)
    {
        for (int i = 0; i < n; ++i)
        {
            Move &m = moves[i];
            int score;
            if (m.from == tt_from && m.to == tt_to && m.promo == tt_promo)
                score = 2000000000;
            else if (m.captured)
                score = 1000000 + 10 * PIECE_VALUES[abs_val(m.captured)] - PIECE_VALUES[abs_val(m.piece)] + (m.promo ? 900 : 0);
            else if (m.promo)
                score = 1000000 + 900;
            else
                score = 0;
            m.score = score;
        }
        // insertion sort (move counts are small)
        for (int i = 1; i < n; ++i)
        {
            Move key = moves[i];
            int j = i - 1;
            while (j >= 0 && moves[j].score < key.score)
            {
                moves[j + 1] = moves[j];
                --j;
            }
            moves[j + 1] = key;
        }
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
                    break; // hit board edge
                if (!pos.board[t])
                    continue; // empty square, keep sliding
                int pt = abs_val(pos.board[t]);
                // found enemy rook or queen on this ray
                if ((pos.board[t] > 0) == (enemy > 0) && (pt == ROOK || pt == QUEEN))
                    return true;
                break; // blocked by any piece (friend or other enemy)
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
    void make_move(Position &pos, int from, int to, int promo = 0)
    {
        int piece = pos.board[from];
        int captured = pos.board[to];
        int old_ep = pos.ep_sq;
        uint8_t old_castling = pos.castling;
        int fwd = (piece > 0) ? 10 : -10;
        int ep_removed = 0;

        pos.board[to] = promo ? (piece > 0 ? promo : -promo) : piece;
        pos.board[from] = EMPTY;

        pos.ep_sq = 0;
        if (abs_val(piece) == PAWN && abs(to - from) == 20)
            pos.ep_sq = from + fwd;
        else if (abs_val(piece) == PAWN && to == old_ep && (to - from == fwd + 1 || to - from == fwd - 1))
        {
            ep_removed = to - fwd;
            pos.board[ep_removed] = EMPTY;
        }

        // castling: move rook along with the king
        if (abs_val(piece) == KING && abs_val(to - from) == 2)
        {
            if (to == 27)
            {
                pos.board[26] = pos.board[28];
                pos.board[28] = EMPTY;
            } // White O-O
            if (to == 23)
            {
                pos.board[24] = pos.board[21];
                pos.board[21] = EMPTY;
            } // White O-O-O
            if (to == 97)
            {
                pos.board[96] = pos.board[98];
                pos.board[98] = EMPTY;
            } // Black O-O
            if (to == 93)
            {
                pos.board[94] = pos.board[91];
                pos.board[91] = EMPTY;
            } // Black O-O-O
        }

        // update castling rights bitmask
        if (from == 25 || to == 25)
            pos.castling &= ~(1 | 2); // white king moved or captured
        if (from == 95 || to == 95)
            pos.castling &= ~(4 | 8); // black king moved or captured
        if (from == 28 || to == 28)
            pos.castling &= ~1; // white K-side rook moved or captured
        if (from == 21 || to == 21)
            pos.castling &= ~2; // white Q-side rook moved or captured
        if (from == 98 || to == 98)
            pos.castling &= ~4; // black K-side rook moved or captured
        if (from == 91 || to == 91)
            pos.castling &= ~8; // black Q-side rook moved or captured

        toggle_move_hash(pos, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);
    }

    // parse move in algebraic notation (e.g., "e2e4", "e7e8q")
    // returns true if valid
    bool parse_move(const char *str, int *from, int *to, int *promo = nullptr)
    {
        if (!str)
            return false;

        // skip leading whitespace
        while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r')
            ++str;

        size_t len = strlen(str);
        if (len != 4 && len != 5)
            return false;

        char f = str[0], r1 = str[1], t = str[2], r2 = str[3];
        *from = sq(f, r1);
        *to = sq(t, r2);
        if (*from == -1 || *to == -1)
            return false;

        if (promo)
        {
            *promo = 0;
            if (len == 5)
            {
                switch (str[4])
                {
                case 'q':
                case 'Q':
                    *promo = QUEEN;
                    break;
                case 'r':
                case 'R':
                    *promo = ROOK;
                    break;
                case 'b':
                case 'B':
                    *promo = BISHOP;
                    break;
                case 'n':
                case 'N':
                    *promo = KNIGHT;
                    break;
                default:
                    return false; // invalid promotion piece
                }
            }
        }
        return true;
    }

    // load a position from a FEN string; returns the side to move (WHITE/BLACK)
    int set_fen(Position &pos, const char *fen)
    {
        for (int i = 0; i < BOARD_SIZE; ++i)
            pos.board[i] = OFF_BOARD;
        for (int r = 2; r <= 9; ++r)
            for (int c = 1; c <= 8; ++c)
                pos.board[r * 10 + c] = EMPTY;
        pos.castling = 0;
        pos.ep_sq = 0;
        pos.best_source = 0;
        pos.best_dest = 0;
        pos.best_promo = 0;

        const char *p = fen;
        int row = 9, col = 1;
        while (*p && *p != ' ')
        {
            char c = *p++;
            if (c == '/')
            {
                --row;
                col = 1;
            }
            else if (c >= '1' && c <= '8')
            {
                col += c - '0';
            }
            else
            {
                int pt = 0;
                switch (c)
                {
                case 'P':
                case 'p':
                    pt = PAWN;
                    break;
                case 'N':
                case 'n':
                    pt = KNIGHT;
                    break;
                case 'B':
                case 'b':
                    pt = BISHOP;
                    break;
                case 'R':
                case 'r':
                    pt = ROOK;
                    break;
                case 'Q':
                case 'q':
                    pt = QUEEN;
                    break;
                case 'K':
                case 'k':
                    pt = KING;
                    break;
                }
                if (pt)
                    pos.board[row * 10 + col] = (c >= 'A' && c <= 'Z') ? pt : -pt;
                ++col;
            }
        }

        if (*p == ' ')
            ++p;
        int side = (*p == 'b') ? BLACK : WHITE;
        if (*p)
            ++p;

        if (*p == ' ')
            ++p;
        while (*p && *p != ' ')
        {
            char c = *p++;
            if (c == 'K')
                pos.castling |= 1;
            if (c == 'Q')
                pos.castling |= 2;
            if (c == 'k')
                pos.castling |= 4;
            if (c == 'q')
                pos.castling |= 8;
        }

        if (*p == ' ')
            ++p;
        if (*p && *p != '-' && *(p + 1) && *(p + 1) != ' ')
        {
            int ep = sq(*p, *(p + 1));
            if (ep != -1)
                pos.ep_sq = ep;
        }

        compute_hash(pos);

        return side;
    }

    // check if a move is legal for the given side
    // verifies piece geometry (pseudo-legality) and king safety
    bool is_legal_move(Position &pos, int side, int from, int to, int promo = 0)
    {
        // basic sanity checks: inside board limits and piece ownership
        if (from < 21 || from > 98 || to < 21 || to > 98)
            return false;
        int piece = pos.board[from];
        int target = pos.board[to];

        if (piece == EMPTY || piece == OFF_BOARD || (piece > 0) != (side > 0))
            return false;
        if (target == OFF_BOARD || (target != EMPTY && (target > 0) == (side > 0)))
            return false;

        int type = abs_val(piece);
        int fwd = (side == WHITE) ? 10 : -10;
        bool pseudo_legal = false;
        bool skip_generic_safety_check = false; // <-- declare here

        // piece geometry and trajectory validation
        if (type == PAWN)
        {
            // single square push
            if (to == from + fwd && target == EMPTY)
            {
                pseudo_legal = true;
            }
            // double square push from starting rank
            else if (to == from + 2 * fwd && target == EMPTY && pos.board[from + fwd] == EMPTY)
            {
                bool at_start = (side == WHITE) ? (from < 40) : (from > 80);
                if (at_start)
                    pseudo_legal = true;
            }
            // diagonal captures (including en passant)
            else if ((to == from + fwd - 1 || to == from + fwd + 1) &&
                     ((target != EMPTY && (target > 0) != (side > 0)) || to == pos.ep_sq))
            {
                pseudo_legal = true;
            }
        }
        else if (type == KNIGHT)
        {
            for (int offset : KNIGHT_OFFSETS)
                if (from + offset == to)
                {
                    pseudo_legal = true;
                    break;
                }
        }
        else // bishop, rook, queen, king
        {
            const std::array<int, 8> *dirs = &KING_OFFSETS;
            int start_dir = 0, end_dir = 8;
            if (type == ROOK)
            {
                start_dir = 0;
                end_dir = 4;
            }
            else if (type == BISHOP)
            {
                start_dir = 4;
                end_dir = 8;
            }
            bool is_slider = (type != KING);

            for (int i = start_dir; i < end_dir; ++i)
            {
                int step = (*dirs)[i];
                int curr = from;
                while (true)
                {
                    curr += step;
                    if (curr == to)
                    {
                        pseudo_legal = true;
                        break;
                    }
                    if (pos.board[curr] != EMPTY || !is_slider)
                        break;
                }
                if (pseudo_legal)
                    break;
            }
        }
        if (type == PAWN && (to / 10 == 2 || to / 10 == 9))
        {
            if (promo == 0 ||
                (promo != QUEEN && promo != ROOK && promo != BISHOP && promo != KNIGHT))
                return false;
        }
        else if (promo != 0)
        {
            return false;
        }

        // Castling validation: king moving two squares
        if (type == KING && abs_val(to - from) == 2)
        {
            if (side == WHITE && from == 25)
            {
                if (to == 27 && (pos.castling & 1) && pos.board[26] == EMPTY && pos.board[27] == EMPTY)
                {
                    if (is_in_check(pos, WHITE))
                        return false;

                    // transit square check
                    pos.board[25] = EMPTY;
                    pos.board[26] = KING;
                    bool transit_safe = !is_in_check(pos, WHITE);
                    pos.board[25] = KING;
                    pos.board[26] = EMPTY;

                    if (transit_safe)
                    {
                        // final check with rook moved
                        pos.board[25] = EMPTY;
                        pos.board[27] = KING;
                        pos.board[28] = EMPTY;
                        pos.board[26] = ROOK;

                        bool final_safe = !is_in_check(pos, WHITE);

                        pos.board[25] = KING;
                        pos.board[27] = EMPTY;
                        pos.board[28] = ROOK;
                        pos.board[26] = EMPTY;

                        if (final_safe)
                        {
                            pseudo_legal = true;
                            skip_generic_safety_check = true;
                        }
                    }
                }
                else if (to == 23 && (pos.castling & 2) && pos.board[24] == EMPTY &&
                         pos.board[23] == EMPTY && pos.board[22] == EMPTY)
                {
                    if (is_in_check(pos, WHITE))
                        return false;

                    pos.board[25] = EMPTY;
                    pos.board[24] = KING;
                    bool transit_safe = !is_in_check(pos, WHITE);
                    pos.board[25] = KING;
                    pos.board[24] = EMPTY;

                    if (transit_safe)
                    {
                        pos.board[25] = EMPTY;
                        pos.board[23] = KING;
                        pos.board[21] = EMPTY;
                        pos.board[24] = ROOK;

                        bool final_safe = !is_in_check(pos, WHITE);

                        pos.board[25] = KING;
                        pos.board[23] = EMPTY;
                        pos.board[21] = ROOK;
                        pos.board[24] = EMPTY;

                        if (final_safe)
                        {
                            pseudo_legal = true;
                            skip_generic_safety_check = true;
                        }
                    }
                }
            }
            else if (side == BLACK && from == 95)
            {
                if (to == 97 && (pos.castling & 4) && pos.board[96] == EMPTY && pos.board[97] == EMPTY)
                {
                    if (is_in_check(pos, BLACK))
                        return false;

                    pos.board[95] = EMPTY;
                    pos.board[96] = -KING;
                    bool transit_safe = !is_in_check(pos, BLACK);
                    pos.board[95] = -KING;
                    pos.board[96] = EMPTY;

                    if (transit_safe)
                    {
                        pos.board[95] = EMPTY;
                        pos.board[97] = -KING;
                        pos.board[98] = EMPTY;
                        pos.board[96] = -ROOK;

                        bool final_safe = !is_in_check(pos, BLACK);

                        pos.board[95] = -KING;
                        pos.board[97] = EMPTY;
                        pos.board[98] = -ROOK;
                        pos.board[96] = EMPTY;

                        if (final_safe)
                        {
                            pseudo_legal = true;
                            skip_generic_safety_check = true;
                        }
                    }
                }
                else if (to == 93 && (pos.castling & 8) && pos.board[94] == EMPTY &&
                         pos.board[93] == EMPTY && pos.board[92] == EMPTY)
                {
                    if (is_in_check(pos, BLACK))
                        return false;

                    pos.board[95] = EMPTY;
                    pos.board[94] = -KING;
                    bool transit_safe = !is_in_check(pos, BLACK);
                    pos.board[95] = -KING;
                    pos.board[94] = EMPTY;

                    if (transit_safe)
                    {
                        pos.board[95] = EMPTY;
                        pos.board[93] = -KING;
                        pos.board[91] = EMPTY;
                        pos.board[94] = -ROOK;

                        bool final_safe = !is_in_check(pos, BLACK);

                        pos.board[95] = -KING;
                        pos.board[93] = EMPTY;
                        pos.board[91] = -ROOK;
                        pos.board[94] = EMPTY;

                        if (final_safe)
                        {
                            pseudo_legal = true;
                            skip_generic_safety_check = true;
                        }
                    }
                }
            }
        }

        if (!pseudo_legal)
            return false;

        // King safety check (generic, for non‑castling moves)
        if (!skip_generic_safety_check)
        {
            int ep_captured = 0;
            pos.board[to] = piece;
            pos.board[from] = EMPTY;
            if (type == PAWN && to == pos.ep_sq && (to - from == fwd + 1 || to - from == fwd - 1))
            {
                ep_captured = to - fwd;
                pos.board[ep_captured] = EMPTY;
            }

            bool leaves_in_check = is_in_check(pos, side);

            pos.board[from] = piece;
            pos.board[to] = target;
            if (ep_captured)
                pos.board[ep_captured] = -PAWN * side;

            return !leaves_in_check;
        }

        // If we reach here, the move was castling and it passed all checks inside the castling block.
        return true;
    }

    // Null move pruning guard: both sides must have a piece beyond king/pawns,
    // otherwise a zugzwang-prone endgame could be misjudged.
    bool has_non_pawn_material(const Position &pos, int side)
    {
        for (int i = 21; i < 99; ++i)
        {
            int piece = pos.board[i];
            if (piece == EMPTY || piece == OFF_BOARD || (piece > 0) != (side > 0))
                continue;
            int type = abs_val(piece);
            if (type > PAWN && type != KING)
                return true;
        }
        return false;
    }

    // AI move using the search function. Iterative deepening with an optional
    // time budget in milliseconds (0 = unlimited, search to max_depth).
    void ai_move(Position &pos, int side, int max_depth, int time_ms,
                 int &from, int &to, int &promo)
    {
        int best_from = 0, best_to = 0, best_promo = 0;
        g_nodes = 0;
        g_timeout = false;
        g_timed = (time_ms > 0);
        if (g_timed)
            g_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_ms);

        // Iterative deepening loop
        for (int d = 1; d <= max_depth; ++d)
        {
            pos.best_source = pos.best_dest = pos.best_promo = 0;
            search(pos, side, d, -30000, 30000, 0);

            if (g_timeout)
                break;

            if (pos.best_source)
            {
                best_from = pos.best_source;
                best_to = pos.best_dest;
                best_promo = pos.best_promo;
            }
        }

        // ------------------------------------------------------------
        // FALLBACK: ensure we always have a legal move if one exists.
        // This can happen if timeout occurred before depth 1 finished,
        // or if some unforeseen condition prevented the root search
        // from recording any move.
        // ------------------------------------------------------------
        if (best_from == 0)
        {
            // First try a non‑timed 1‑ply search (forces a move to be found).
            bool saved_timed = g_timed;
            bool saved_timeout = g_timeout;

            g_timed = false;
            g_timeout = false;

            pos.best_source = pos.best_dest = pos.best_promo = 0;
            search(pos, side, 1, -30000, 30000, 0);

            if (pos.best_source)
            {
                best_from = pos.best_source;
                best_to = pos.best_dest;
                best_promo = pos.best_promo;
            }

            g_timed = saved_timed;
            g_timeout = saved_timeout;

            // If even that fails (shouldn't happen for a non‑terminal position),
            // manually generate and test moves.
            if (best_from == 0)
            {
                Move moves[256];
                int n = 0;
                generate_moves(pos, side, moves, n, false);

                for (int i = 0; i < n; ++i)
                {
                    int ep_removed = 0;
                    uint8_t old_castling = pos.castling;
                    int old_ep = pos.ep_sq;

                    if (apply_move(pos, side, moves[i].from, moves[i].to,
                                   moves[i].piece, moves[i].captured, moves[i].promo,
                                   ep_removed, old_castling, old_ep))
                    {
                        // Legal move found – undo and record it.
                        undo_move(pos, side, moves[i].from, moves[i].to,
                                  moves[i].piece, moves[i].captured, moves[i].promo,
                                  ep_removed, old_castling, old_ep);

                        best_from = moves[i].from;
                        best_to = moves[i].to;
                        best_promo = moves[i].promo;
                        break;
                    }
                }
            }
        }

        from = best_from;
        to = best_to;
        promo = best_promo;
    }

} // namespace chess

// convert 120-square index to algebraic notation
void square_to_algebraic(int sq, char *buf)
{
    int file = (sq % 10) - 1;
    int rank = (sq / 10) - 2;
    buf[0] = 'a' + file;
    buf[1] = '1' + rank;
    buf[2] = '\0';
}

int main()
{
    chess::init_zobrist();
    chess::Position pos;
    pos.init();
    int side = chess::WHITE;

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line == "uci")
        {
            std::cout << "id name ChessEngine\n";
            std::cout << "id author You\n";
            std::cout << "uciok\n"
                      << std::flush;
        }
        else if (line == "isready")
        {
            std::cout << "readyok\n"
                      << std::flush;
        }
        else if (line == "ucinewgame")
        {
            pos.init();
            side = chess::WHITE;
            chess::tt_clear();
        }
        else if (line.rfind("position", 0) == 0)
        {
            size_t moves_pos = line.find(" moves ");
            std::string pos_part = (moves_pos == std::string::npos) ? line : line.substr(0, moves_pos);

            if (pos_part.compare(0, 9, "position ") == 0)
            {
                std::string rest = pos_part.substr(9);
                if (rest.rfind("startpos", 0) == 0)
                {
                    pos.init();
                    side = chess::WHITE;
                }
                else if (rest.rfind("fen ", 0) == 0)
                {
                    side = chess::set_fen(pos, rest.c_str() + 4);
                }
            }

            if (moves_pos != std::string::npos)
            {
                std::stringstream ss(line.substr(moves_pos + 7));
                std::string move_str;
                while (ss >> move_str)
                {
                    int from, to, promo;
                    // Inside the "position ... moves" processing
                    if (chess::parse_move(move_str.c_str(), &from, &to, &promo) &&
                        chess::is_legal_move(pos, side, from, to, promo))
                    {
                        chess::make_move(pos, from, to, promo);
                        side = -side;
                    }
                }
            }
        }
        else if (line.rfind("go", 0) == 0)
        {
            int depth = 5;
            size_t dpos = line.find("depth");
            if (dpos != std::string::npos)
            {
                depth = std::atoi(line.c_str() + dpos + 6);
                if (depth < 1)
                    depth = 1;
                if (depth > 20)
                    depth = 20;
            }

            // Parse the clock parameters (only used when no explicit depth is given).
            int time_ms = 0;
            if (dpos == std::string::npos)
            {
                int wtime = 0, btime = 0, winc = 0, binc = 0, mtg = 0;
                std::stringstream ss(line.substr(3));
                std::string tok;
                while (ss >> tok)
                {
                    if (tok == "wtime")
                        ss >> wtime;
                    else if (tok == "btime")
                        ss >> btime;
                    else if (tok == "winc")
                        ss >> winc;
                    else if (tok == "binc")
                        ss >> binc;
                    else if (tok == "movestogo")
                        ss >> mtg;
                }
                int clock = (side == chess::WHITE) ? wtime : btime;
                int inc = (side == chess::WHITE) ? winc : binc;
                if (clock > 0)
                {
                    // under time control the engine should search as deep as the
                    // budget allows, so raise the iterative deepening cap
                    depth = 64;
                    int alloc = mtg ? clock / mtg : clock / 30;
                    alloc += inc / 2;
                    if (alloc > clock / 2)
                        alloc = clock / 2;
                    if (alloc < 100)
                        alloc = 100;
                    time_ms = alloc;
                }
            }

            int from, to, promo;
            ai_move(pos, side, depth, time_ms, from, to, promo);

            if (from == 0)
            {
                std::cout << "bestmove 0000\n"
                          << std::flush;
            }
            else
            {
                char from_alg[3], to_alg[3];
                square_to_algebraic(from, from_alg);
                square_to_algebraic(to, to_alg);
                std::cout << "bestmove " << from_alg << to_alg;
                if (promo)
                {
                    char pc = (promo == chess::QUEEN) ? 'q' : (promo == chess::ROOK) ? 'r'
                                                          : (promo == chess::BISHOP) ? 'b'
                                                                                     : 'n';
                    std::cout << pc;
                }
                std::cout << "\n"
                          << std::flush;
            }
        }
        else if (line == "quit")
        {
            break;
        }
    }
    return 0;
}