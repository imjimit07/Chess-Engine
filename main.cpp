#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>

namespace chess
{
    // =========================================================================
    // Global State & Game History
    // =========================================================================
    std::vector<uint64_t> rep_stack;        // Search repetition history stack
    std::vector<uint64_t> g_game_history;   // Whole game position history for draw detection
    std::vector<std::string> g_played_moves; // UCI move list for the current position
    bool g_is_startpos = true;              // Whether position was loaded from startpos
    int g_age = 0;                          // Transposition table search generation counter

    // =========================================================================
    // Opening Book (Sequences of standard algebraic UCI moves)
    // =========================================================================
    const std::vector<std::vector<std::string>> OPENING_BOOK = {
        // ===== Flank & English Openings =====
        {"c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6", "e2e3", "f8b4", "d2d4", "e5e4"},
        {"c2c4", "c7c5", "g1f3", "g8f6", "b1c3", "b8c6", "g2g3", "d7d5", "c4d5", "f6d5", "f1g2"},
        {"g1f3", "d7d5", "c2c4", "e7e6", "g2g3", "g8f6", "f1g2", "f8e7", "e1g1", "e8g8", "b2b3"},
        {"g1f3", "d7d5", "g2g3", "g8f6", "f1g2", "e7e6", "e1g1", "f8e7", "d2d3", "e8g8"},

        // ===== 1.d4 Queen's Pawn Openings =====
        {"d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5", "f8e7", "e2e3", "e8g8", "g1f3"},
        {"d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4", "a2a4", "c8f5", "e2e3"},
        {"d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8", "f1d3", "d7d5", "g1f3"},
        {"d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3", "e8g8", "f1e2"},
        {"d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6", "b1c3", "e6d5", "c4d5", "d7d6", "g1f3"},
        {"d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "e7e6", "e2e3", "f8d6", "f4g3", "e8g8", "f1d3"},

        // ===== 1.e4 King's Pawn Openings =====
        {"e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d3", "d7d6", "e1g1"},
        {"e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7", "f1e1"},
        {"e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4", "f8c5", "c1e3", "d8f6", "c2c3"},
        {"e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3"},
        {"e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "e7e5", "d4b5"},
        {"e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c1g5", "f8e7", "e4e5", "f6d7", "g5e7"},
        {"e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4", "c8f5", "e4g3", "f5g6", "h2h4"},
        {"e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4", "g8f6", "g1f3", "c7c6", "c1d2"}
    };

    // =========================================================================
    // Board Representation Constants
    // =========================================================================
    constexpr int BOARD_SIZE = 120; // 10x12 mailbox board representation
    constexpr int EMPTY = 0;
    constexpr int OFF_BOARD = 7;

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

    // Base material values in centipawns
    constexpr std::array<int, 7> PIECE_VALUES = {0, 100, 320, 330, 500, 950, 20000};

    // Move offsets on the 10x12 board
    constexpr std::array<int, 8> KNIGHT_OFFSETS = {-21, -19, -12, -8, 8, 12, 19, 21};
    constexpr std::array<int, 8> KING_OFFSETS   = {-1, 1, -10, 10, -11, -9, 9, 11};

    // Search limits and mate thresholds
    constexpr int MAX_PLY = 256;
    constexpr int MATE = 30000;
    constexpr int MATE_THRESHOLD = 29000;

    // =========================================================================
    // Piece-Square Tables (PST) - Oriented from Rank 1 (Row 0) to Rank 8 (Row 7)
    // Symmetrical for White; mirrored for Black.
    // =========================================================================

    // Pawn PST (Middlegame & Endgame)
    constexpr std::array<int, 64> PAWN_MG_PST = {
          0,   0,   0,   0,   0,   0,   0,   0,
          5,  10,  10, -20, -20,  10,  10,   5,
          5,  -5, -10,   0,   0, -10,  -5,   5,
          0,   0,   0,  20,  20,   0,   0,   0,
          5,   5,  10,  25,  25,  10,   5,   5,
         10,  10,  20,  30,  30,  20,  10,  10,
         50,  50,  50,  50,  50,  50,  50,  50,
          0,   0,   0,   0,   0,   0,   0,   0
    };
    constexpr std::array<int, 64> PAWN_EG_PST = {
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          5,   5,  10,  15,  15,  10,   5,   5,
         10,  10,  15,  20,  20,  15,  10,  10,
         20,  20,  25,  30,  30,  25,  20,  20,
         35,  35,  40,  45,  45,  40,  35,  35,
         60,  60,  60,  60,  60,  60,  60,  60,
          0,   0,   0,   0,   0,   0,   0,   0
    };

    // Knight PST (Middlegame & Endgame)
    constexpr std::array<int, 64> KNIGHT_MG_PST = {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    };
    constexpr std::array<int, 64> KNIGHT_EG_PST = {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    };

    // Bishop PST (Middlegame & Endgame)
    constexpr std::array<int, 64> BISHOP_MG_PST = {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    };
    constexpr std::array<int, 64> BISHOP_EG_PST = {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    };

    // Rook PST (Middlegame & Endgame) - 7th rank reward is on Row 6 (Rank 7)
    constexpr std::array<int, 64> ROOK_MG_PST = {
          0,   0,   0,   5,   5,   0,   0,   0,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          5,  10,  10,  10,  10,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0
    };
    constexpr std::array<int, 64> ROOK_EG_PST = {
          0,   0,   0,   0,   0,   0,   0,   0,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          5,  10,  10,  10,  10,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0
    };

    // Queen PST (Middlegame & Endgame)
    constexpr std::array<int, 64> QUEEN_MG_PST = {
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -10,   5,   5,   5,   5,   5,   0, -10,
          0,   0,   5,   5,   5,   5,   0,  -5,
         -5,   0,   5,   5,   5,   5,   0,  -5,
        -10,   0,   5,   5,   5,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20
    };
    constexpr std::array<int, 64> QUEEN_EG_PST = {
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -10,   0,   5,   5,   5,   5,   0, -10,
         -5,   0,   5,   5,   5,   5,   0,  -5,
          0,   0,   5,   5,   5,   5,   0,  -5,
        -10,   5,   5,   5,   5,   5,   0, -10,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20
    };

    // King Middlegame: Rewards castled King on g1/c1/b1, penalizes wandering into the center
    constexpr std::array<int, 64> KING_MG_PST = {
         20,  30,  10,   0,   0,  10,  30,  20,
         20,  20,   0,   0,   0,   0,  20,  20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30
    };

    // King Endgame: Rewards centralized, active king
    constexpr std::array<int, 64> KING_EG_PST = {
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10,   0,   0, -10, -20, -30,
        -30, -10,  20,  30,  30,  20, -10, -30,
        -30, -10,  30,  40,  40,  30, -10, -30,
        -30, -10,  30,  40,  40,  30, -10, -30,
        -30, -10,  20,  30,  30,  20, -10, -30,
        -30, -30,   0,   0,   0,   0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    };

    inline int abs_val(int x) { return x < 0 ? -x : x; }

    // =========================================================================
    // Move Struct & Principal Variation (PV) Table
    // =========================================================================
    struct Move
    {
        int from = 0;
        int to = 0;
        int promo = 0;
        int captured = 0;
        int piece = 0;
        int score = 0;
    };

    constexpr int PV_MAX_PLY = 256;
    Move pv_table[PV_MAX_PLY][PV_MAX_PLY];
    int pv_length[PV_MAX_PLY];

    // =========================================================================
    // Search Heuristics Tables
    // =========================================================================
    int killer_moves[MAX_PLY][2];          // 2 Killer moves per ply (packed as from*1000 + to)
    int history_table[7][BOARD_SIZE];       // History heuristic [piece_type][to_sq]

    // Time management state
    bool g_timed = false;
    bool g_timeout = false;
    long long g_nodes = 0;
    std::chrono::steady_clock::time_point g_deadline;

    // =========================================================================
    // Position Data Structure
    // =========================================================================
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
        uint64_t hash = 0;         // Zobrist hash of pieces + castling + ep

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
                    int back_piece = "42356324"[col - 1] - '0';
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

    // =========================================================================
    // Transposition Table (Zobrist Hashing)
    // =========================================================================
    enum TTFlag : uint8_t
    {
        TT_EXACT = 0,
        TT_LOWER = 1,
        TT_UPPER = 2
    };

    struct TTEntry
    {
        uint64_t key = 0;
        int score = 0;
        int depth = -1;
        uint8_t flag = 0;
        int from = 0;
        int to = 0;
        int promo = 0;
        int age = 0;
    };

    std::vector<TTEntry> g_tt;
    int g_tt_mask = 0;

    uint64_t g_zpiece[7][BOARD_SIZE]; // Zobrist piece keys [type][square]
    uint64_t g_zcastling[16];         // Zobrist castling keys
    uint64_t g_zep[9];                // Zobrist en-passant file keys (0 = none)
    uint64_t g_zside;                 // Zobrist side to move key

    uint64_t splitmix64(uint64_t &x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        uint64_t z = x;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    void tt_clear()
    {
        for (std::size_t i = 0; i < g_tt.size(); ++i)
            g_tt[i].key = 0;
        g_age = 0;
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

        g_tt.assign(1 << 20, TTEntry()); // 1M entries (~32MB)
        g_tt_mask = (1 << 20) - 1;
        tt_clear();
    }

    inline int ep_file(int sq) { return sq ? (sq % 10) : 0; }

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

    void tt_store(uint64_t key, int depth, int score, int flag, int from, int to, int promo, int ply)
    {
        // Adjust mate scores before storing so they reflect distance from the node
        if (score >= MATE_THRESHOLD)
            score += ply;
        else if (score <= -MATE_THRESHOLD)
            score -= ply;

        TTEntry &e = g_tt[key & g_tt_mask];

        // Replacement scheme: prefer deeper or newer search results
        if (e.key == 0 || e.depth <= depth || e.age < g_age)
        {
            e.key = key;
            e.score = score;
            e.depth = depth;
            e.flag = flag;
            e.from = from;
            e.to = to;
            e.promo = promo;
            e.age = g_age;
        }
    }

    // =========================================================================
    // Forward Declarations & Helpers
    // =========================================================================
    bool is_in_check(const Position &pos, int side);
    bool has_non_pawn_material(const Position &pos, int side);
    bool insufficient_material(const Position &pos);
    bool is_endgame(const Position &pos);
    int see(const Position &pos, int from, int to, int side);
    void square_to_algebraic(int sq, char *buf);

    inline bool pawn_start_rank(int side, int sq)
    {
        return (side == WHITE) ? (sq / 10 == 3) : (sq / 10 == 8);
    }

    // Lookup opening book move matching current sequence
    std::string get_book_move(const std::vector<std::string> &played_moves)
    {
        std::vector<std::string> candidates;
        for (const auto &line : OPENING_BOOK)
        {
            if (line.size() > played_moves.size())
            {
                bool match = true;
                for (size_t i = 0; i < played_moves.size(); ++i)
                {
                    if (line[i] != played_moves[i])
                    {
                        match = false;
                        break;
                    }
                }
                if (match)
                    candidates.push_back(line[played_moves.size()]);
            }
        }
        if (candidates.empty())
            return "";
        int idx = rand() % candidates.size();
        return candidates[idx];
    }

    // =========================================================================
    // Static Position Evaluation (Tapered Middlegame / Endgame)
    // =========================================================================
    inline int get_pst_value(int type, int row, int col, bool is_endgame_phase)
    {
        int idx = (row - 2) * 8 + (col - 1);
        switch (type)
        {
        case PAWN:
            return is_endgame_phase ? PAWN_EG_PST[idx] : PAWN_MG_PST[idx];
        case KNIGHT:
            return is_endgame_phase ? KNIGHT_EG_PST[idx] : KNIGHT_MG_PST[idx];
        case BISHOP:
            return is_endgame_phase ? BISHOP_EG_PST[idx] : BISHOP_MG_PST[idx];
        case ROOK:
            return is_endgame_phase ? ROOK_EG_PST[idx] : ROOK_MG_PST[idx];
        case QUEEN:
            return is_endgame_phase ? QUEEN_EG_PST[idx] : QUEEN_MG_PST[idx];
        case KING:
            return is_endgame_phase ? KING_EG_PST[idx] : KING_MG_PST[idx];
        default:
            return 0;
        }
    }

    int evaluate_position(const Position &pos, int side)
    {
        int mg_score = 0;
        int eg_score = 0;
        int phase = 0; // Game phase: 0 (endgame) to 24 (opening/middlegame)

        int white_pawn_files[8] = {0};
        int black_pawn_files[8] = {0};
        int white_pawn_mask[8] = {0};
        int black_pawn_mask[8] = {0};
        int white_king = 0, black_king = 0;
        int white_bishops = 0, black_bishops = 0;

        // 1. Material, Piece-Square Tables, and Game Phase
        for (int i = 21; i < 99; ++i)
        {
            int piece = pos.board[i];
            if (piece == EMPTY || piece == OFF_BOARD)
                continue;

            int type = abs_val(piece);
            int row = i / 10;
            int col = i % 10;

            // Phase accumulation (Knight=1, Bishop=1, Rook=2, Queen=4)
            if (type == KNIGHT || type == BISHOP) phase += 1;
            else if (type == ROOK)                phase += 2;
            else if (type == QUEEN)               phase += 4;

            if (piece > 0) // White piece
            {
                mg_score += PIECE_VALUES[type];
                eg_score += PIECE_VALUES[type];

                // PST for White (natural rank row-2)
                mg_score += get_pst_value(type, row, col, false);
                eg_score += get_pst_value(type, row, col, true);

                if (type == PAWN)
                {
                    white_pawn_files[col - 1]++;
                    white_pawn_mask[col - 1] |= (1 << (row - 2));
                }
                else if (type == BISHOP) white_bishops++;
                else if (type == KING)   white_king = i;
            }
            else // Black piece
            {
                mg_score -= PIECE_VALUES[type];
                eg_score -= PIECE_VALUES[type];

                // PST for Black (vertically mirrored rank: 9 - row + 2 = 11 - row)
                int black_row = 11 - row;
                mg_score -= get_pst_value(type, black_row, col, false);
                eg_score -= get_pst_value(type, black_row, col, true);

                if (type == PAWN)
                {
                    black_pawn_files[col - 1]++;
                    black_pawn_mask[col - 1] |= (1 << (row - 2));
                }
                else if (type == BISHOP) black_bishops++;
                else if (type == KING)   black_king = i;
            }
        }

        phase = std::min(phase, 24);

        // 2. Bishop Pair Bonus
        if (white_bishops >= 2) { mg_score += 30; eg_score += 35; }
        if (black_bishops >= 2) { mg_score -= 30; eg_score -= 35; }

        // 3. Pawn Structure Evaluation (Passed, Isolated, Doubled)
        for (int i = 21; i < 99; ++i)
        {
            int piece = pos.board[i];
            if (piece == EMPTY || piece == OFF_BOARD || abs_val(piece) != PAWN)
                continue;

            int row = i / 10;
            int col = i % 10;
            int f = col - 1;

            if (piece > 0) // White pawn
            {
                // Passed pawn check: no enemy pawns on same/adjacent files in front (ranks > row)
                int adj = 0;
                for (int j = std::max(0, f - 1); j <= std::min(7, f + 1); ++j)
                    adj |= black_pawn_mask[j];

                bool passed = ((adj & (0xFF << (row - 1))) == 0);
                if (passed)
                {
                    int rank = row - 2; // 0..7 (rank 1 to 8)
                    int bonus_mg = rank * 6;
                    int bonus_eg = rank * 14;
                    if (rank >= 5) { bonus_mg += 20; bonus_eg += 35; }

                    // King proximity bonus in late endgames
                    if (phase < 8 && white_king && black_king)
                    {
                        int dist_wk = std::max(std::abs(white_king / 10 - row), std::abs(white_king % 10 - col));
                        int dist_bk = std::max(std::abs(black_king / 10 - row), std::abs(black_king % 10 - col));
                        if (dist_wk < dist_bk) bonus_eg += 20;
                    }

                    mg_score += bonus_mg;
                    eg_score += bonus_eg;
                }

                // Isolated pawn penalty
                bool isolated = (f > 0 ? white_pawn_files[f - 1] : 0) == 0 &&
                                (f < 7 ? white_pawn_files[f + 1] : 0) == 0;
                if (isolated)
                {
                    mg_score -= 10;
                    eg_score -= 15;
                }
            }
            else // Black pawn
            {
                int adj = 0;
                for (int j = std::max(0, f - 1); j <= std::min(7, f + 1); ++j)
                    adj |= white_pawn_mask[j];

                bool passed = ((adj & ((1 << (row - 2)) - 1)) == 0);
                if (passed)
                {
                    int rank = 9 - row; // 0..7 (rank 8 to 1 from Black's view)
                    int bonus_mg = rank * 6;
                    int bonus_eg = rank * 14;
                    if (rank >= 5) { bonus_mg += 20; bonus_eg += 35; }

                    if (phase < 8 && white_king && black_king)
                    {
                        int dist_bk = std::max(std::abs(black_king / 10 - row), std::abs(black_king % 10 - col));
                        int dist_wk = std::max(std::abs(white_king / 10 - row), std::abs(white_king % 10 - col));
                        if (dist_bk < dist_wk) bonus_eg += 20;
                    }

                    mg_score -= bonus_mg;
                    eg_score -= bonus_eg;
                }

                bool isolated = (f > 0 ? black_pawn_files[f - 1] : 0) == 0 &&
                                (f < 7 ? black_pawn_files[f + 1] : 0) == 0;
                if (isolated)
                {
                    mg_score += 10;
                    eg_score += 15;
                }
            }
        }

        // Doubled pawns penalty
        for (int f = 0; f < 8; ++f)
        {
            if (white_pawn_files[f] > 1)
            {
                int extra = white_pawn_files[f] - 1;
                mg_score -= extra * 12;
                eg_score -= extra * 15;
            }
            if (black_pawn_files[f] > 1)
            {
                int extra = black_pawn_files[f] - 1;
                mg_score -= extra * 12;
                eg_score -= extra * 15;
            }
        }

        // 4. Rook on Open / Semi-open files
        for (int i = 21; i < 99; ++i)
        {
            int p = pos.board[i];
            if (p == ROOK)
            {
                int f = (i % 10) - 1;
                if (white_pawn_files[f] == 0)
                {
                    if (black_pawn_files[f] == 0) { mg_score += 20; eg_score += 15; } // Open file
                    else                          { mg_score += 10; eg_score += 10; } // Semi-open file
                }
            }
            else if (p == -ROOK)
            {
                int f = (i % 10) - 1;
                if (black_pawn_files[f] == 0)
                {
                    if (white_pawn_files[f] == 0) { mg_score -= 20; eg_score -= 15; }
                    else                          { mg_score -= 10; eg_score -= 10; }
                }
            }
        }

        // 5. King Safety Pawn Shield in Middlegame
        if (phase > 12)
        {
            if (white_king)
            {
                int kcol = white_king % 10;
                int krow = white_king / 10;
                if (krow <= 3 && kcol >= 6) // White King castled kingside
                {
                    if (pos.board[36] != PAWN && pos.board[46] != PAWN) mg_score -= 15; // f-file shield
                    if (pos.board[37] != PAWN && pos.board[47] != PAWN) mg_score -= 20; // g-file shield
                    if (pos.board[38] != PAWN && pos.board[48] != PAWN) mg_score -= 10; // h-file shield
                }
                else if (krow <= 3 && kcol <= 3) // White King castled queenside
                {
                    if (pos.board[31] != PAWN && pos.board[41] != PAWN) mg_score -= 10;
                    if (pos.board[32] != PAWN && pos.board[42] != PAWN) mg_score -= 20;
                    if (pos.board[33] != PAWN && pos.board[43] != PAWN) mg_score -= 15;
                }
            }

            if (black_king)
            {
                int kcol = black_king % 10;
                int krow = black_king / 10;
                if (krow >= 8 && kcol >= 6) // Black King castled kingside
                {
                    if (pos.board[86] != -PAWN && pos.board[76] != -PAWN) mg_score += 15;
                    if (pos.board[87] != -PAWN && pos.board[77] != -PAWN) mg_score += 20;
                    if (pos.board[88] != -PAWN && pos.board[78] != -PAWN) mg_score += 10;
                }
                else if (krow >= 8 && kcol <= 3) // Black King castled queenside
                {
                    if (pos.board[81] != -PAWN && pos.board[71] != -PAWN) mg_score += 10;
                    if (pos.board[82] != -PAWN && pos.board[72] != -PAWN) mg_score += 20;
                    if (pos.board[83] != -PAWN && pos.board[73] != -PAWN) mg_score += 15;
                }
            }
        }

        // 6. Minor Piece Development Bonus in Opening / Early Middlegame
        if (phase > 18)
        {
            if (pos.board[22] == KNIGHT) mg_score -= 10;
            if (pos.board[23] == BISHOP) mg_score -= 10;
            if (pos.board[26] == BISHOP) mg_score -= 10;
            if (pos.board[27] == KNIGHT) mg_score -= 10;

            if (pos.board[92] == -KNIGHT) mg_score += 10;
            if (pos.board[93] == -BISHOP) mg_score += 10;
            if (pos.board[96] == -BISHOP) mg_score += 10;
            if (pos.board[97] == -KNIGHT) mg_score += 10;
        }

        // 7. Tempo Bonus
        mg_score += 10;
        eg_score += 10;

        // Tapered Interpolation between Middlegame and Endgame
        int interpolated = (mg_score * phase + eg_score * (24 - phase)) / 24;

        return interpolated * side;
    }

    // =========================================================================
    // Static Exchange Evaluation (SEE)
    // Correct minimax-folded simulation of piece exchanges on target square.
    // =========================================================================
    int see(const Position &pos, int from, int to, int side)
    {
        std::array<int, BOARD_SIZE> b = pos.board;

        int gain[32];
        int d = 0;

        int captured = b[to];
        gain[0] = PIECE_VALUES[abs_val(captured)];

        // Simulate initial capture
        int piece_on_target = b[from];
        b[to] = piece_on_target;
        b[from] = EMPTY;

        int current_side = -side;

        while (d < 31)
        {
            int best_from = 0;
            int best_val = 1000000;

            for (int sq = 21; sq < 99; ++sq)
            {
                int p = b[sq];
                if (p == EMPTY || p == OFF_BOARD || (p > 0) != (current_side > 0))
                    continue;

                int pt = abs_val(p);
                int val = PIECE_VALUES[pt];

                bool attacks = false;
                if (pt == PAWN)
                {
                    int fwd = (current_side == WHITE) ? 10 : -10;
                    attacks = (to == sq + fwd - 1 || to == sq + fwd + 1);
                }
                else if (pt == KNIGHT)
                {
                    for (int off : KNIGHT_OFFSETS)
                    {
                        if (sq + off == to) { attacks = true; break; }
                    }
                }
                else if (pt == KING)
                {
                    for (int off : KING_OFFSETS)
                    {
                        if (sq + off == to) { attacks = true; break; }
                    }
                }
                else // Bishop, Rook, Queen
                {
                    int start = (pt == ROOK) ? 0 : (pt == BISHOP ? 4 : 0);
                    int end = (pt == ROOK) ? 4 : (pt == BISHOP ? 8 : 8);

                    for (int dir = start; dir < end; ++dir)
                    {
                        int step = KING_OFFSETS[dir];
                        int t = sq;
                        while (true)
                        {
                            t += step;
                            if (b[t] == OFF_BOARD) break;
                            if (t == to) { attacks = true; break; }
                            if (b[t] != EMPTY) break;
                        }
                        if (attacks) break;
                    }
                }

                if (attacks && val < best_val)
                {
                    best_from = sq;
                    best_val = val;
                }
            }

            if (!best_from)
                break;

            ++d;
            gain[d] = PIECE_VALUES[abs_val(piece_on_target)] - gain[d - 1];

            piece_on_target = b[best_from];
            b[to] = piece_on_target;
            b[best_from] = EMPTY;

            current_side = -current_side;
        }

        // Minimax folding back from leaf
        while (--d > 0)
        {
            gain[d - 1] = -std::max(-gain[d - 1], gain[d]);
        }

        return gain[0];
    }

    // =========================================================================
    // Move Generation & Make / Undo Move
    // =========================================================================
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
            int rfrom = (to == 27) ? 28 : (to == 23) ? 21 : (to == 97) ? 98 : 91;
            int rto   = (to == 27) ? 26 : (to == 23) ? 24 : (to == 97) ? 96 : 94;
            pos.hash ^= g_zpiece[ROOK][rfrom] ^ g_zpiece[ROOK][rto];
        }

        if (pos.castling != old_castling)
            pos.hash ^= g_zcastling[old_castling] ^ g_zcastling[pos.castling];
        pos.hash ^= g_zep[ep_file(old_ep)] ^ g_zep[ep_file(pos.ep_sq)];
    }

    bool apply_move(Position &pos, int side, int from, int to, int piece, int captured, int promo,
                    int &ep_removed, uint8_t &old_castling, int &old_ep)
    {
        old_castling = pos.castling;
        old_ep = pos.ep_sq;
        int fwd = (side == WHITE) ? 10 : -10;
        bool castle_move = (abs_val(piece) == KING && abs(to - from) == 2);
        ep_removed = 0;

        // Cannot castle while currently in check
        if (castle_move && is_in_check(pos, side))
            return false;

        // Castling transit square safety check
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

        // Make the move on the board
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

        // Relocate rook for castling
        if (castle_move)
        {
            if (to == 27)      { pos.board[26] = pos.board[28]; pos.board[28] = EMPTY; } // White O-O
            else if (to == 23) { pos.board[24] = pos.board[21]; pos.board[21] = EMPTY; } // White O-O-O
            else if (to == 97) { pos.board[96] = pos.board[98]; pos.board[98] = EMPTY; } // Black O-O
            else if (to == 93) { pos.board[94] = pos.board[91]; pos.board[91] = EMPTY; } // Black O-O-O
        }

        // Update castling rights
        if (from == 25 || to == 25) pos.castling &= ~(1 | 2); // White King moved / captured
        if (from == 95 || to == 95) pos.castling &= ~(4 | 8); // Black King moved / captured
        if (from == 28 || to == 28) pos.castling &= ~1;       // White h1 Rook
        if (from == 21 || to == 21) pos.castling &= ~2;       // White a1 Rook
        if (from == 98 || to == 98) pos.castling &= ~4;       // Black h8 Rook
        if (from == 91 || to == 91) pos.castling &= ~8;       // Black a8 Rook

        toggle_move_hash(pos, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);

        // Disallow moves that leave the moving side's king in check
        if (is_in_check(pos, side))
        {
            toggle_move_hash(pos, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);
            pos.board[from] = piece;
            pos.board[to] = ep_removed ? EMPTY : captured;
            if (ep_removed) pos.board[ep_removed] = -PAWN * side;
            if (castle_move)
            {
                if (to == 27)      { pos.board[28] = pos.board[26]; pos.board[26] = EMPTY; }
                else if (to == 23) { pos.board[21] = pos.board[24]; pos.board[24] = EMPTY; }
                else if (to == 97) { pos.board[98] = pos.board[96]; pos.board[96] = EMPTY; }
                else if (to == 93) { pos.board[91] = pos.board[94]; pos.board[94] = EMPTY; }
            }
            pos.castling = old_castling;
            pos.ep_sq = old_ep;
            return false;
        }

        return true;
    }

    void undo_move(Position &pos, int side, int from, int to, int piece, int captured, int promo,
                   int ep_removed, uint8_t old_castling, int old_ep)
    {
        toggle_move_hash(pos, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);

        bool castle_move = (abs_val(piece) == KING && abs(to - from) == 2);
        pos.board[from] = piece;
        pos.board[to] = ep_removed ? EMPTY : captured;
        if (ep_removed)
            pos.board[ep_removed] = -PAWN * side;

        if (castle_move)
        {
            if (to == 27)      { pos.board[28] = pos.board[26]; pos.board[26] = EMPTY; }
            else if (to == 23) { pos.board[21] = pos.board[24]; pos.board[24] = EMPTY; }
            else if (to == 97) { pos.board[98] = pos.board[96]; pos.board[96] = EMPTY; }
            else if (to == 93) { pos.board[91] = pos.board[94]; pos.board[94] = EMPTY; }
        }

        pos.castling = old_castling;
        pos.ep_sq = old_ep;
    }

    inline void add_move(Move *moves, int &n, int from, int to, int promo, int captured, int piece)
    {
        if (n >= 256) return;
        moves[n].from = from;
        moves[n].to = to;
        moves[n].promo = promo;
        moves[n].captured = captured;
        moves[n].piece = piece;
        moves[n].score = 0;
        ++n;
    }

    void generate_moves(const Position &pos, int side, Move *moves, int &n, bool captures_only)
    {
        int fwd = (side == WHITE) ? 10 : -10;
        int promo_rank = (side == WHITE) ? 9 : 2;

        for (int from = 21; from < 99; ++from)
        {
            int piece = pos.board[from];
            if (piece == OFF_BOARD || piece == EMPTY || (piece > 0) != (side > 0))
                continue;

            int type = abs_val(piece);

            if (type == PAWN)
            {
                // Diagonal Captures (including en passant)
                for (int dx = -1; dx <= 1; dx += 2)
                {
                    int to = from + fwd + dx;
                    int target = pos.board[to];
                    bool ep = (to == pos.ep_sq) && (target == EMPTY);
                    int cap_piece = ep ? (-PAWN * side) : target;

                    if (ep || (target != EMPTY && target != OFF_BOARD && (target > 0) != (side > 0)))
                    {
                        if (to / 10 == promo_rank)
                        {
                            for (int pr : {QUEEN, ROOK, BISHOP, KNIGHT})
                                add_move(moves, n, from, to, pr, cap_piece, piece);
                        }
                        else
                        {
                            add_move(moves, n, from, to, 0, cap_piece, piece);
                        }
                    }
                }

                // Quiet Pushes
                int to = from + fwd;
                bool is_promo_push = (to / 10 == promo_rank);
                if ((!captures_only || is_promo_push) && pos.board[to] == EMPTY)
                {
                    if (is_promo_push)
                    {
                        for (int pr : {QUEEN, ROOK, BISHOP, KNIGHT})
                            add_move(moves, n, from, to, pr, 0, piece);
                    }
                    else
                    {
                        add_move(moves, n, from, to, 0, 0, piece);
                    }

                    // Double square push from starting rank
                    if (!captures_only && pawn_start_rank(side, from) && pos.board[from + 2 * fwd] == EMPTY)
                    {
                        add_move(moves, n, from, from + 2 * fwd, 0, 0, piece);
                    }
                }
            }
            else
            {
                int start_dir = (type == ROOK) ? 0 : (type == BISHOP ? 4 : 0);
                int end_dir   = (type == ROOK) ? 4 : (type == BISHOP ? 8 : 8);
                const std::array<int, 8> *dirs = (type == KNIGHT) ? &KNIGHT_OFFSETS : &KING_OFFSETS;
                bool is_slider = (type != KNIGHT && type != KING);

                for (int i = start_dir; i < end_dir; ++i)
                {
                    int step = (*dirs)[i];
                    int to = from;
                    while (true)
                    {
                        to += step;
                        int target = pos.board[to];
                        if (target == OFF_BOARD) break;
                        if (target && (target > 0) == (side > 0)) break;

                        if (target)
                        {
                            add_move(moves, n, from, to, 0, target, piece);
                            break;
                        }

                        if (!captures_only)
                            add_move(moves, n, from, to, 0, 0, piece);

                        if (!is_slider) break;
                    }
                }

                // Castling generation (quiet moves only)
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

    // =========================================================================
    // Move Ordering (TT Move -> Winning Captures / MVV-LVA -> Killers -> History)
    // =========================================================================
    void order_moves(Move *moves, int n, int tt_from, int tt_to, int tt_promo, int ply)
    {
        for (int i = 0; i < n; ++i)
        {
            Move &m = moves[i];

            if (m.from == tt_from && m.to == tt_to && m.promo == tt_promo)
            {
                m.score = 2000000000; // 1. Transposition Table Best Move
            }
            else if (m.captured)
            {
                int victim_val = PIECE_VALUES[abs_val(m.captured)];
                int attacker_val = PIECE_VALUES[abs_val(m.piece)];
                int mvv_lva = victim_val * 10 - attacker_val;

                if (m.promo == QUEEN)
                    m.score = 1500000 + mvv_lva;
                else
                    m.score = 1000000 + mvv_lva;
            }
            else if (m.promo == QUEEN)
            {
                m.score = 950000; // Queen promotion push
            }
            else
            {
                int move_key = m.from * 1000 + m.to;
                if (move_key == killer_moves[ply][0])
                    m.score = 900000;
                else if (move_key == killer_moves[ply][1])
                    m.score = 800000;
                else
                    m.score = std::min(history_table[abs_val(m.piece)][m.to], 700000);
            }
        }

        // Fast insertion sort
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

    // =========================================================================
    // Check Detection & Game State Checks
    // =========================================================================
    bool is_in_check(const Position &pos, int side)
    {
        int king_sq = 0;
        int enemy = -side;

        for (int i = 21; i < 99; ++i)
        {
            if (pos.board[i] == KING * side)
            {
                king_sq = i;
                break;
            }
        }
        if (!king_sq) return false;

        // Pawn attacks
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

        // Knight attacks
        for (int i = 0; i < 8; ++i)
            if (pos.board[king_sq + KNIGHT_OFFSETS[i]] == KNIGHT * enemy)
                return true;

        // King attacks
        for (int i = 0; i < 8; ++i)
            if (pos.board[king_sq + KING_OFFSETS[i]] == KING * enemy)
                return true;

        // Orthogonal sliders (Rooks & Queens)
        for (int i = 0; i < 4; ++i)
        {
            int t = king_sq;
            while (true)
            {
                t += KING_OFFSETS[i];
                if (pos.board[t] == OFF_BOARD) break;
                if (!pos.board[t]) continue;
                int pt = abs_val(pos.board[t]);
                if ((pos.board[t] > 0) == (enemy > 0) && (pt == ROOK || pt == QUEEN))
                    return true;
                break;
            }
        }

        // Diagonal sliders (Bishops & Queens)
        for (int i = 4; i < 8; ++i)
        {
            int t = king_sq;
            while (true)
            {
                t += KING_OFFSETS[i];
                if (pos.board[t] == OFF_BOARD) break;
                if (!pos.board[t]) continue;
                int pt = abs_val(pos.board[t]);
                if ((pos.board[t] > 0) == (enemy > 0) && (pt == BISHOP || pt == QUEEN))
                    return true;
                break;
            }
        }

        return false;
    }

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

    bool insufficient_material(const Position &pos)
    {
        int knights = 0, bishops = 0, rooks = 0, queens = 0, pawns = 0;

        for (int i = 21; i < 99; ++i)
        {
            int p = pos.board[i];
            if (p == EMPTY || p == OFF_BOARD) continue;
            int type = abs_val(p);
            if (type == PAWN)        pawns++;
            else if (type == KNIGHT) knights++;
            else if (type == BISHOP) bishops++;
            else if (type == ROOK)   rooks++;
            else if (type == QUEEN)  queens++;
        }

        if (pawns > 0 || rooks > 0 || queens > 0)
            return false;

        // K vs K, K+N vs K, K+B vs K
        if (knights + bishops <= 1)
            return true;

        // K+B vs K+B with bishops on same color
        if (knights == 0 && bishops == 2)
        {
            int wb = 0, bb = 0, wb_col = -1, bb_col = -1;
            for (int i = 21; i < 99; ++i)
            {
                int p = pos.board[i];
                if (p == BISHOP)
                {
                    wb++;
                    wb_col = (i / 10 + i % 10) % 2;
                }
                else if (p == -BISHOP)
                {
                    bb++;
                    bb_col = (i / 10 + i % 10) % 2;
                }
            }
            if (wb == 1 && bb == 1 && wb_col == bb_col)
                return true;
        }

        return false;
    }

    bool is_endgame(const Position &pos)
    {
        int queens = 0, rooks = 0;
        for (int i = 21; i < 99; ++i)
        {
            int p = pos.board[i];
            if (p == QUEEN || p == -QUEEN) queens++;
            else if (p == ROOK || p == -ROOK) rooks++;
        }
        return (queens == 0 && rooks <= 2);
    }

    // =========================================================================
    // Quiescence Search
    // Resolves capture chains and tactical volatility at leaf nodes.
    // =========================================================================
    int quiesce(Position &pos, int side, int alpha, int beta, int ply)
    {
        if (g_timeout || (++g_nodes % 2048 == 0 && g_timed && std::chrono::steady_clock::now() >= g_deadline))
        {
            g_timeout = true;
            return alpha;
        }

        if (ply >= MAX_PLY - 1)
            return evaluate_position(pos, side);

        bool in_check = is_in_check(pos, side);

        int stand_pat = 0;
        if (!in_check)
        {
            stand_pat = evaluate_position(pos, side);
            if (stand_pat >= beta)
                return beta;
            if (stand_pat > alpha)
                alpha = stand_pat;
        }

        Move moves[256];
        int n = 0;
        generate_moves(pos, side, moves, n, !in_check);

        for (int i = 0; i < n; ++i)
        {
            if (moves[i].captured)
            {
                int victim_val = PIECE_VALUES[abs_val(moves[i].captured)];
                int attacker_val = PIECE_VALUES[abs_val(moves[i].piece)];
                moves[i].score = 1000000 + victim_val * 10 - attacker_val;
            }
            else if (moves[i].promo == QUEEN)
            {
                moves[i].score = 950000;
            }
            else
            {
                moves[i].score = 0;
            }
        }

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

        int legal_moves = 0;
        for (int i = 0; i < n; ++i)
        {
            // Delta pruning
            if (!in_check && moves[i].captured && !moves[i].promo)
            {
                int cap_val = PIECE_VALUES[abs_val(moves[i].captured)];
                if (stand_pat + cap_val + 200 < alpha)
                    continue;
            }

            // SEE pruning: skip captures that lose material
            if (!in_check && moves[i].captured && see(pos, moves[i].from, moves[i].to, side) < 0)
                continue;

            int ep_removed = 0;
            uint8_t old_castling = pos.castling;
            int old_ep = pos.ep_sq;

            if (!apply_move(pos, side, moves[i].from, moves[i].to, moves[i].piece,
                            moves[i].captured, moves[i].promo, ep_removed, old_castling, old_ep))
                continue;

            legal_moves++;
            int score = -quiesce(pos, -side, -beta, -alpha, ply + 1);
            undo_move(pos, side, moves[i].from, moves[i].to, moves[i].piece,
                      moves[i].captured, moves[i].promo, ep_removed, old_castling, old_ep);

            if (g_timeout)
                return alpha;

            if (score >= beta)
                return beta;
            if (score > alpha)
                alpha = score;
        }

        if (in_check && legal_moves == 0)
            return -MATE + ply;

        return alpha;
    }

    // =========================================================================
    // Principal Variation Search (Alpha-Beta with PVS, NMP, and LMR)
    // =========================================================================
    int search(Position &pos, int side, int depth, int alpha, int beta, int ply, bool null_move_allowed = true)
    {
        pv_length[ply] = 0;

        if (g_timeout || (++g_nodes % 2048 == 0 && g_timed && std::chrono::steady_clock::now() >= g_deadline))
        {
            g_timeout = true;
            return alpha;
        }

        if (ply >= MAX_PLY - 1)
            return evaluate_position(pos, side);

        if (depth <= 0)
            return quiesce(pos, side, alpha, beta, ply);

        // Repetition & Draw Detection
        if (ply > 0)
        {
            uint64_t rep_key = pos.hash ^ (side == WHITE ? 0 : g_zside);
            if (rep_stack.size() > 1)
            {
                for (size_t i = 0; i + 1 < rep_stack.size(); ++i)
                {
                    if (rep_stack[i] == rep_key)
                        return 0; // Draw by repetition
                }
            }
            if (insufficient_material(pos))
                return 0;
        }

        bool in_check = is_in_check(pos, side);

        // Check extension
        if (in_check)
            depth++;

        uint64_t key = pos.hash ^ (side == WHITE ? 0 : g_zside);

        // Transposition Table Probe
        int tt_from = 0, tt_to = 0, tt_promo = 0;
        const TTEntry *e = &g_tt[key & g_tt_mask];
        if (e->key == key)
        {
            tt_from = e->from;
            tt_to = e->to;
            tt_promo = e->promo;

            if (ply > 0 && e->depth >= depth)
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

        int static_eval = evaluate_position(pos, side);

        // Null Move Pruning (NMP)
        if (null_move_allowed && !in_check && depth >= 3 && ply > 0 &&
            has_non_pawn_material(pos, side) && static_eval >= beta)
        {
            int R = (depth >= 6) ? 3 : 2;
            int saved_ep = pos.ep_sq;
            pos.hash ^= g_zep[ep_file(saved_ep)] ^ g_zep[0];
            pos.ep_sq = 0;

            int null_score = -search(pos, -side, depth - 1 - R, -beta, -beta + 1, ply + 1, false);

            pos.ep_sq = saved_ep;
            pos.hash ^= g_zep[ep_file(saved_ep)] ^ g_zep[0];

            if (null_score >= beta)
                return (null_score >= MATE_THRESHOLD) ? beta : null_score;
        }

        Move moves[256];
        int n = 0;
        generate_moves(pos, side, moves, n, false);

        if (n == 0)
        {
            if (in_check)
                return -MATE + ply; // Checkmate
            return 0;               // Stalemate
        }

        order_moves(moves, n, tt_from, tt_to, tt_promo, ply);

        int orig_alpha = alpha;
        int best_score = -1000000;
        int best_from = 0, best_to = 0, best_promo = 0;
        int legal_moves = 0;

        for (int i = 0; i < n; ++i)
        {
            int from = moves[i].from, to = moves[i].to, promo = moves[i].promo;
            int piece = moves[i].piece, captured = moves[i].captured;

            int ep_removed = 0;
            uint8_t old_castling = pos.castling;
            int old_ep = pos.ep_sq;

            if (!apply_move(pos, side, from, to, piece, captured, promo, ep_removed, old_castling, old_ep))
                continue;

            legal_moves++;

            uint64_t child_side_bit = (-side == WHITE) ? 0 : g_zside;
            rep_stack.push_back(pos.hash ^ child_side_bit);

            int score;

            if (legal_moves == 1)
            {
                // PV move: search with full window
                score = -search(pos, -side, depth - 1, -beta, -alpha, ply + 1);
            }
            else
            {
                // Late Move Reductions (LMR)
                bool is_tactical = (captured != 0 || promo != 0 || in_check || is_in_check(pos, -side));
                bool is_killer = (from * 1000 + to == killer_moves[ply][0] || from * 1000 + to == killer_moves[ply][1]);
                bool is_tt = (from == tt_from && to == tt_to && promo == tt_promo);

                int reduction = 0;
                if (depth >= 3 && legal_moves >= 4 && !is_tactical && !is_killer && !is_tt)
                {
                    reduction = 1;
                    if (depth >= 6 && legal_moves >= 8)
                        reduction = 2;
                }

                int child_depth = std::max(0, depth - 1 - reduction);

                // Zero window search (PVS)
                score = -search(pos, -side, child_depth, -alpha - 1, -alpha, ply + 1);

                if (reduction > 0 && score > alpha)
                {
                    score = -search(pos, -side, depth - 1, -alpha - 1, -alpha, ply + 1);
                }

                if (score > alpha && score < beta)
                {
                    score = -search(pos, -side, depth - 1, -beta, -alpha, ply + 1);
                }
            }

            rep_stack.pop_back();
            undo_move(pos, side, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);

            if (g_timeout)
                return alpha;

            if (score > best_score)
            {
                best_score = score;
                best_from = from;
                best_to = to;
                best_promo = promo;
            }

            if (score >= beta)
            {
                if (!captured && !promo)
                {
                    int move_key = from * 1000 + to;
                    killer_moves[ply][1] = killer_moves[ply][0];
                    killer_moves[ply][0] = move_key;

                    history_table[abs_val(piece)][to] += depth * depth;
                    if (history_table[abs_val(piece)][to] > 1000000)
                        history_table[abs_val(piece)][to] = 1000000;
                }

                tt_store(key, depth, score, TT_LOWER, from, to, promo, ply);
                return beta;
            }

            if (score > alpha)
            {
                alpha = score;

                // Update Principal Variation table
                pv_table[ply][ply].from = from;
                pv_table[ply][ply].to = to;
                pv_table[ply][ply].promo = promo;
                pv_table[ply][ply].captured = captured;
                pv_table[ply][ply].piece = piece;

                for (int k = 0; k < pv_length[ply + 1]; ++k)
                    pv_table[ply][ply + 1 + k] = pv_table[ply + 1][ply + 1 + k];
                pv_length[ply] = pv_length[ply + 1] + 1;

                if (ply == 0)
                {
                    pos.best_source = from;
                    pos.best_dest = to;
                    pos.best_promo = promo;
                }
            }
        }

        if (legal_moves == 0)
        {
            if (in_check)
                return -MATE + ply;
            return 0;
        }

        if (alpha > orig_alpha)
            tt_store(key, depth, alpha, TT_EXACT, best_from, best_to, best_promo, ply);
        else
            tt_store(key, depth, alpha, TT_UPPER, 0, 0, 0, ply);

        return alpha;
    }

    // =========================================================================
    // Iterative Deepening & Root AI Move
    // =========================================================================
    void ai_move(Position &pos, int side, int max_depth, int time_ms,
                 int &from, int &to, int &promo)
    {
        int best_from = 0, best_to = 0, best_promo = 0;
        g_nodes = 0;
        g_timeout = false;
        g_timed = (time_ms > 0);
        if (g_timed)
            g_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(time_ms);

        auto start_time = std::chrono::steady_clock::now();
        g_age++;

        // Query opening book if game started from startpos
        if (g_is_startpos && !g_played_moves.empty())
        {
            std::string book_move = get_book_move(g_played_moves);
            if (!book_move.empty())
            {
                int b_from, b_to, b_promo = 0;
                char b_from_f = book_move[0], b_from_r = book_move[1];
                char b_to_f = book_move[2], b_to_r = book_move[3];
                b_from = (b_from_r - '0' + 1) * 10 + (b_from_f - 'a' + 1);
                b_to = (b_to_r - '0' + 1) * 10 + (b_to_f - 'a' + 1);
                if (book_move.length() == 5)
                {
                    char p = book_move[4];
                    if (p == 'q') b_promo = QUEEN;
                    else if (p == 'r') b_promo = ROOK;
                    else if (p == 'b') b_promo = BISHOP;
                    else if (p == 'n') b_promo = KNIGHT;
                }

                Move moves[256];
                int n = 0;
                generate_moves(pos, side, moves, n, false);
                for (int i = 0; i < n; ++i)
                {
                    if (moves[i].from == b_from && moves[i].to == b_to && moves[i].promo == b_promo)
                    {
                        int ep_removed = 0;
                        uint8_t old_castling = pos.castling;
                        int old_ep = pos.ep_sq;
                        if (apply_move(pos, side, b_from, b_to, moves[i].piece, moves[i].captured, b_promo,
                                       ep_removed, old_castling, old_ep))
                        {
                            undo_move(pos, side, b_from, b_to, moves[i].piece, moves[i].captured, b_promo,
                                      ep_removed, old_castling, old_ep);
                            from = b_from;
                            to = b_to;
                            promo = b_promo;
                            return;
                        }
                    }
                }
            }
        }

        int prev_score = 0;

        for (int d = 1; d <= max_depth; ++d)
        {
            memset(killer_moves, 0, sizeof(killer_moves));

            int alpha = -30000;
            int beta = 30000;

            if (d >= 4)
            {
                alpha = prev_score - 50;
                beta = prev_score + 50;
            }

            pos.best_source = pos.best_dest = pos.best_promo = 0;
            for (int i = 0; i < PV_MAX_PLY; ++i)
                pv_length[i] = 0;

            rep_stack = g_game_history;

            int score = search(pos, side, d, alpha, beta, 0);

            // Aspiration window failure: re-search with full alpha-beta bounds
            if (!g_timeout && (score <= alpha || score >= beta))
            {
                alpha = -30000;
                beta = 30000;
                pos.best_source = pos.best_dest = pos.best_promo = 0;
                rep_stack = g_game_history;
                score = search(pos, side, d, alpha, beta, 0);
            }

            if (g_timeout)
                break;

            prev_score = score;

            if (pos.best_source)
            {
                best_from = pos.best_source;
                best_to = pos.best_dest;
                best_promo = pos.best_promo;
            }

            // UCI info output
            auto now = std::chrono::steady_clock::now();
            long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            long long nps = (elapsed_ms > 0) ? (g_nodes * 1000 / elapsed_ms) : 0;

            std::string pv_str;
            int pv_len = std::min(pv_length[0], 20);
            for (int k = 0; k < pv_len; ++k)
            {
                if (pv_table[0][k].from == 0 && pv_table[0][k].to == 0)
                    break;
                char from_alg[3], to_alg[3];
                square_to_algebraic(pv_table[0][k].from, from_alg);
                square_to_algebraic(pv_table[0][k].to, to_alg);
                pv_str += std::string(from_alg) + to_alg;
                if (pv_table[0][k].promo)
                {
                    char pc = (pv_table[0][k].promo == QUEEN) ? 'q' :
                              (pv_table[0][k].promo == ROOK) ? 'r' :
                              (pv_table[0][k].promo == BISHOP) ? 'b' : 'n';
                    pv_str += pc;
                }
                pv_str += " ";
            }

            std::cout << "info depth " << d
                      << " score cp " << score
                      << " nodes " << g_nodes
                      << " nps " << nps
                      << " time " << elapsed_ms
                      << " pv " << pv_str << "\n"
                      << std::flush;

            if (score >= MATE_THRESHOLD || score <= -MATE_THRESHOLD)
                break;
        }

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
                if (apply_move(pos, side, moves[i].from, moves[i].to, moves[i].piece,
                               moves[i].captured, moves[i].promo, ep_removed, old_castling, old_ep))
                {
                    undo_move(pos, side, moves[i].from, moves[i].to, moves[i].piece,
                              moves[i].captured, moves[i].promo, ep_removed, old_castling, old_ep);
                    best_from = moves[i].from;
                    best_to = moves[i].to;
                    best_promo = moves[i].promo;
                    break;
                }
            }
        }

        from = best_from;
        to = best_to;
        promo = best_promo;
    }

    // =========================================================================
    // Algebraic Notation Parsing & Formatting
    // =========================================================================
    void square_to_algebraic(int sq, char *buf)
    {
        int file = (sq % 10) - 1;
        int rank = (sq / 10) - 2;
        buf[0] = 'a' + file;
        buf[1] = '1' + rank;
        buf[2] = '\0';
    }

    int sq(char file, char rank)
    {
        if (file < 'a' || file > 'h' || rank < '1' || rank > '8')
            return -1;
        return (rank - '0' + 1) * 10 + (file - 'a' + 1);
    }

    bool parse_move(const char *str, int *from, int *to, int *promo = nullptr)
    {
        if (!str) return false;

        while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r')
            ++str;

        size_t len = strlen(str);
        while (len > 0 && (str[len - 1] == ' ' || str[len - 1] == '\t' ||
                           str[len - 1] == '\n' || str[len - 1] == '\r'))
            --len;

        if (len != 4 && len != 5)
            return false;

        *from = sq(str[0], str[1]);
        *to = sq(str[2], str[3]);
        if (*from == -1 || *to == -1)
            return false;

        if (promo)
        {
            *promo = 0;
            if (len == 5)
            {
                switch (str[4])
                {
                case 'q': case 'Q': *promo = QUEEN; break;
                case 'r': case 'R': *promo = ROOK; break;
                case 'b': case 'B': *promo = BISHOP; break;
                case 'n': case 'N': *promo = KNIGHT; break;
                default: return false;
                }
            }
        }
        return true;
    }

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
                case 'P': case 'p': pt = PAWN; break;
                case 'N': case 'n': pt = KNIGHT; break;
                case 'B': case 'b': pt = BISHOP; break;
                case 'R': case 'r': pt = ROOK; break;
                case 'Q': case 'q': pt = QUEEN; break;
                case 'K': case 'k': pt = KING; break;
                }
                if (pt)
                    pos.board[row * 10 + col] = (c >= 'A' && c <= 'Z') ? pt : -pt;
                ++col;
            }
        }

        if (*p == ' ') ++p;
        int side = (*p == 'b') ? BLACK : WHITE;
        if (*p) ++p;

        if (*p == ' ') ++p;
        while (*p && *p != ' ')
        {
            char c = *p++;
            if (c == 'K') pos.castling |= 1;
            if (c == 'Q') pos.castling |= 2;
            if (c == 'k') pos.castling |= 4;
            if (c == 'q') pos.castling |= 8;
        }

        if (*p == ' ') ++p;
        if (*p && *p != '-' && *(p + 1) && *(p + 1) != ' ')
        {
            int ep = sq(*p, *(p + 1));
            if (ep != -1)
                pos.ep_sq = ep;
        }

        compute_hash(pos);
        return side;
    }

    bool is_legal_move(Position &pos, int side, int from, int to, int promo = 0)
    {
        Move moves[256];
        int n = 0;
        generate_moves(pos, side, moves, n, false);

        for (int i = 0; i < n; ++i)
        {
            if (moves[i].from == from && moves[i].to == to && moves[i].promo == promo)
            {
                int ep_removed = 0;
                uint8_t old_castling = pos.castling;
                int old_ep = pos.ep_sq;
                if (apply_move(pos, side, from, to, moves[i].piece, moves[i].captured, promo,
                               ep_removed, old_castling, old_ep))
                {
                    undo_move(pos, side, from, to, moves[i].piece, moves[i].captured, promo,
                              ep_removed, old_castling, old_ep);
                    return true;
                }
            }
        }
        return false;
    }

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

        if (abs_val(piece) == KING && abs_val(to - from) == 2)
        {
            if (to == 27)      { pos.board[26] = pos.board[28]; pos.board[28] = EMPTY; }
            else if (to == 23) { pos.board[24] = pos.board[21]; pos.board[21] = EMPTY; }
            else if (to == 97) { pos.board[96] = pos.board[98]; pos.board[98] = EMPTY; }
            else if (to == 93) { pos.board[94] = pos.board[91]; pos.board[91] = EMPTY; }
        }

        if (from == 25 || to == 25) pos.castling &= ~(1 | 2);
        if (from == 95 || to == 95) pos.castling &= ~(4 | 8);
        if (from == 28 || to == 28) pos.castling &= ~1;
        if (from == 21 || to == 21) pos.castling &= ~2;
        if (from == 98 || to == 98) pos.castling &= ~4;
        if (from == 91 || to == 91) pos.castling &= ~8;

        toggle_move_hash(pos, from, to, piece, captured, promo, ep_removed, old_castling, old_ep);
    }

} // namespace chess

// =============================================================================
// Main UCI Loop
// =============================================================================
int main()
{
    std::srand(static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count()));
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
            std::cout << "id author Antigravity\n";
            std::cout << "uciok\n" << std::flush;
        }
        else if (line == "isready")
        {
            std::cout << "readyok\n" << std::flush;
        }
        else if (line == "ucinewgame")
        {
            pos.init();
            side = chess::WHITE;
            chess::tt_clear();
            memset(chess::history_table, 0, sizeof(chess::history_table));
            memset(chess::killer_moves, 0, sizeof(chess::killer_moves));
            chess::g_played_moves.clear();
            chess::g_game_history.clear();
            chess::g_is_startpos = true;
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
                    chess::g_is_startpos = true;
                }
                else if (rest.rfind("fen ", 0) == 0)
                {
                    side = chess::set_fen(pos, rest.c_str() + 4);
                    chess::g_is_startpos = false;
                }
            }

            chess::g_played_moves.clear();
            chess::g_game_history.clear();
            chess::g_game_history.push_back(pos.hash ^ (side == chess::WHITE ? 0 : chess::g_zside));

            if (moves_pos != std::string::npos)
            {
                std::stringstream ss(line.substr(moves_pos + 7));
                std::string move_str;
                while (ss >> move_str)
                {
                    int from, to, promo;
                    if (chess::parse_move(move_str.c_str(), &from, &to, &promo) &&
                        chess::is_legal_move(pos, side, from, to, promo))
                    {
                        chess::make_move(pos, from, to, promo);
                        side = -side;
                        chess::g_played_moves.push_back(move_str);
                        chess::g_game_history.push_back(pos.hash ^ (side == chess::WHITE ? 0 : chess::g_zside));
                    }
                }
            }
        }
        else if (line.rfind("go", 0) == 0)
        {
            int depth = 64; // Default to iterative deepening with clock limit
            size_t dpos = line.find("depth");
            if (dpos != std::string::npos)
            {
                depth = std::atoi(line.c_str() + dpos + 6);
                if (depth < 1) depth = 1;
                if (depth > 64) depth = 64;
            }

            int time_ms = 0;
            if (dpos == std::string::npos)
            {
                int wtime = 0, btime = 0, winc = 0, binc = 0, mtg = 0;
                std::stringstream ss(line.substr(3));
                std::string tok;
                while (ss >> tok)
                {
                    if (tok == "wtime") ss >> wtime;
                    else if (tok == "btime") ss >> btime;
                    else if (tok == "winc") ss >> winc;
                    else if (tok == "binc") ss >> binc;
                    else if (tok == "movestogo") ss >> mtg;
                }

                int clock = (side == chess::WHITE) ? wtime : btime;
                int inc = (side == chess::WHITE) ? winc : binc;

                if (clock > 0)
                {
                    int alloc = mtg ? (clock / mtg) : (clock / 28);
                    alloc += inc / 2;
                    if (alloc > clock / 2) alloc = clock / 2;
                    if (alloc < 50) alloc = 50;
                    time_ms = alloc;
                }
                else
                {
                    depth = 6; // Default fixed depth if no time or depth specified
                }
            }

            int from = 0, to = 0, promo = 0;
            chess::ai_move(pos, side, depth, time_ms, from, to, promo);

            if (from == 0)
            {
                std::cout << "bestmove 0000\n" << std::flush;
            }
            else
            {
                char from_alg[3], to_alg[3];
                chess::square_to_algebraic(from, from_alg);
                chess::square_to_algebraic(to, to_alg);
                std::cout << "bestmove " << from_alg << to_alg;
                if (promo)
                {
                    char pc = (promo == chess::QUEEN) ? 'q' :
                              (promo == chess::ROOK) ? 'r' :
                              (promo == chess::BISHOP) ? 'b' : 'n';
                    std::cout << pc;
                }
                std::cout << "\n" << std::flush;
            }
        }
        else if (line == "quit")
        {
            break;
        }
    }

    return 0;
}
