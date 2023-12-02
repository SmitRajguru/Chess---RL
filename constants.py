from enum import Enum
from pygame import mixer

mixer.init()

PieceType = Enum("pieceType", ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"])

ColorType = Enum("colorType", ["WHITE", "BLACK", "EMPTY"])

MoveType = Enum(
    "moveType",
    [
        "VALID",
        "INVALID",
        "CAPTURE",
        "CASTLE",
        "PROMOTION",
        "CHECK",
        "CHECKMATE",
        "STALEMATE",
    ],
)

PieceValues = {
    PieceType.PAWN: 1,
    PieceType.KNIGHT: 3,
    PieceType.BISHOP: 3.25,
    PieceType.ROOK: 5,
    PieceType.QUEEN: 9,
    PieceType.KING: 100,
}

PieceSymbols = {
    ".": (PieceType.PAWN, ColorType.EMPTY),
    "P": (PieceType.PAWN, ColorType.WHITE),
    "N": (PieceType.KNIGHT, ColorType.WHITE),
    "B": (PieceType.BISHOP, ColorType.WHITE),
    "R": (PieceType.ROOK, ColorType.WHITE),
    "Q": (PieceType.QUEEN, ColorType.WHITE),
    "K": (PieceType.KING, ColorType.WHITE),
    "p": (PieceType.PAWN, ColorType.BLACK),
    "n": (PieceType.KNIGHT, ColorType.BLACK),
    "b": (PieceType.BISHOP, ColorType.BLACK),
    "r": (PieceType.ROOK, ColorType.BLACK),
    "q": (PieceType.QUEEN, ColorType.BLACK),
    "k": (PieceType.KING, ColorType.BLACK),
}

action2moveMap = {}


action2moveMap[ColorType.WHITE] = []
action2moveMap[ColorType.BLACK] = []

for start in range(64):
    for end in range(64):
        action = f""
        action += chr(97 + start % 8)
        action += str(8 - start // 8)
        action += chr(97 + end % 8)
        action += str(8 - end // 8)
        # print(f"white: {action} for {start} to {end}")
        action2moveMap[ColorType.WHITE].append(action)

        action = f""
        action += chr(97 + start % 8)
        action += str(1 + start // 8)
        action += chr(97 + end % 8)
        action += str(1 + end // 8)
        # print(f"black: {action} for {start} to {end}")
        action2moveMap[ColorType.BLACK].append(action)
