// Header files needed

#include <memory.h>
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dos.h>
#include <time.h>

// DEFINE values

#define U64 unsigned __int64
#define BITBOARD unsigned __int64

// Chess Squares

#define A1 0
#define B1 1
#define C1 2
#define D1 3
#define E1 4
#define F1 5
#define G1 6
#define H1 7
#define A2 8
#define B2 9
#define C2 10
#define D2 11
#define E2 12
#define F2 13
#define G2 14
#define H2 15
#define A3 16
#define B3 17
#define C3 18
#define D3 19
#define E3 20
#define F3 21
#define G3 22
#define H3 23
#define A4 24
#define B4 25
#define C4 26
#define D4 27
#define E4 28
#define F4 29
#define G4 30
#define H4 31
#define A5 32
#define B5 33
#define C5 34
#define D5 35
#define E5 36
#define F5 37
#define G5 38
#define H5 39
#define A6 40
#define B6 41
#define C6 42
#define D6 43
#define E6 44
#define F6 45
#define G6 46
#define H6 47
#define A7 48
#define B7 49
#define C7 50
#define D7 51
#define E7 52
#define F7 53
#define G7 54
#define H7 55
#define A8 56
#define B8 57
#define C8 58
#define D8 59
#define E8 60
#define F8 61
#define G8 62
#define H8 63

// Move Directions (in which direction a piece can move)

#define NORTH 0
#define NE 1
#define EAST 2
#define SE 3
#define SOUTH 4
#define SW 5
#define WEST 6
#define NW 7

// Pieces

#define P 0 // pawn
#define N 1 // knight
#define B 2 // bishop
#define R 3 // rook
#define Q 4 // queen
#define K 5 // king
#define EMPTY 6

// Pieces Colors

#define White 0
#define Black 1

// Number of moves searched
#define MAX_PLY 64
#define MOVE_STACK 4000 // if the search exceeds this valuse, increase it.

// Game List
#define GAME_STACK 2000
#define HASH_SCORE 100000000 // added to the move score so that the move from the hash table is searched first
#define CAPTURE_SCORE 10000000 // added to the move score so that captures are searched after the move from the hash table
typedef struct
{
    int start;
    int dest;
    int promote; // if the piece is a pawn
    int score; // moves with higher score will be searched first
} move;
typedef struct
{
    int start;
    int dest;
    int promote;
    int capture;
    int fifty; // keeps track of how many moves since the last pawn move or capture
    int castle;

    // test for repitiion
    U64 hash;
    U64 lock;
} game;
game game_list[GAME_STACK];
