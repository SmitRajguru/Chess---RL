from constants import *
import chess
import gymnasium as gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import time

print("Loading Chess Environment...")
time.sleep(1)

plt.ion()


rewardType = Enum(
    "rewardType",
    [
        "INVALID_MOVE",
        "VALID_MOVE",
        "CAPTURE",
        "PROMOTION",
        "WIN",
        "LOSS",
        "DRAW",
        "CHECK",
    ],
)

rewards = {
    rewardType.INVALID_MOVE: -10,
    rewardType.VALID_MOVE: 0.1,
    rewardType.CAPTURE: 0.5,
    rewardType.PROMOTION: 0.5,
    rewardType.WIN: 10,
    rewardType.LOSS: -10,
    rewardType.DRAW: 0,
    rewardType.CHECK: 1,
}


# chess environment for reinforcement learning
class ChessEnv(gym.Env):
    def __init__(self, resolution=100, margin=0.1):
        self.board = chess.Board()
        self.state = self.board2state()
        self.valueBoard = self.state2valueBoard()

        self.action_space = spaces.Discrete(4096)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(14, 8, 8), dtype=np.uint8
        )
        self.count = 0

        self.resolution = resolution
        self.margin = margin
        self.load_images()

        self.mute = False
        self.load_sounds()

        self.fig = None
        self.ax = None

        self.reset()

    # load images
    def load_images(self):
        # Initialize board colors
        self.boardColors = {
            ColorType.WHITE: cv2.imread(
                "assets/pieces/square brown light_png_shadow_1024px.png",
                cv2.IMREAD_UNCHANGED,
            ),
            ColorType.BLACK: cv2.imread(
                "assets/pieces/square brown dark_png_shadow_1024px.png",
                cv2.IMREAD_UNCHANGED,
            ),
        }

        # Initialize piece pieces
        self.pieceImages = {
            (PieceType.PAWN, ColorType.WHITE): cv2.imread(
                f"assets/pieces/w_pawn_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.KNIGHT, ColorType.WHITE): cv2.imread(
                f"assets/pieces/w_knight_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.BISHOP, ColorType.WHITE): cv2.imread(
                f"assets/pieces/w_bishop_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.ROOK, ColorType.WHITE): cv2.imread(
                f"assets/pieces/w_rook_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.QUEEN, ColorType.WHITE): cv2.imread(
                f"assets/pieces/w_queen_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.KING, ColorType.WHITE): cv2.imread(
                f"assets/pieces/w_king_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.PAWN, ColorType.BLACK): cv2.imread(
                f"assets/pieces/b_pawn_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.KNIGHT, ColorType.BLACK): cv2.imread(
                f"assets/pieces/b_knight_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.BISHOP, ColorType.BLACK): cv2.imread(
                f"assets/pieces/b_bishop_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.ROOK, ColorType.BLACK): cv2.imread(
                f"assets/pieces/b_rook_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.QUEEN, ColorType.BLACK): cv2.imread(
                f"assets/pieces/b_queen_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
            (PieceType.KING, ColorType.BLACK): cv2.imread(
                f"assets/pieces/b_king_png_shadow_1024px.png", cv2.IMREAD_UNCHANGED
            ),
        }

        # Resize piece pieces to resolution
        for key in self.boardColors:
            self.boardColors[key] = cv2.resize(
                self.boardColors[key], (self.resolution, self.resolution)
            )
        for key in self.pieceImages:
            self.pieceImages[key] = cv2.resize(
                self.pieceImages[key],
                (
                    int(self.resolution * (1 - 2 * self.margin)),
                    int(self.resolution * (1 - 2 * self.margin)),
                ),
            )

        # convert color pieces to RGB
        for key in self.boardColors:
            self.boardColors[key] = cv2.cvtColor(
                self.boardColors[key], cv2.COLOR_BGRA2RGBA
            )
        for key in self.pieceImages:
            self.pieceImages[key] = cv2.cvtColor(
                self.pieceImages[key], cv2.COLOR_BGRA2RGBA
            )

    # load sounds
    def load_sounds(self):
        # Initialize move sounds
        self.moveSounds = {
            rewardType.INVALID_MOVE: mixer.Sound("assets/sounds/notify.mp3"),
            rewardType.VALID_MOVE: mixer.Sound("assets/sounds/valid.mp3"),
            rewardType.CAPTURE: mixer.Sound("assets/sounds/capture.mp3"),
            rewardType.PROMOTION: mixer.Sound("assets/sounds/notify.mp3"),
            rewardType.WIN: mixer.Sound("assets/sounds/notify.mp3"),
            rewardType.LOSS: mixer.Sound("assets/sounds/notify.mp3"),
            rewardType.DRAW: mixer.Sound("assets/sounds/notify.mp3"),
            rewardType.CHECK: mixer.Sound("assets/sounds/notify.mp3"),
        }

    def reset(self):
        self.board.reset()
        self.state = self.board2state()
        self.valueBoard = self.state2valueBoard()
        self.show_board()
        return self.valueBoard

    def render(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)
            plt.tight_layout()
            plt.show()
        self.show_board()

    def board2state(self):
        boardStr = str(self.board)

        # Convert board string to state
        state = np.zeros((8, 8), dtype=str)

        boardStr = boardStr.split("\n")
        for row in range(8):
            boardStr[row] = boardStr[row].split(" ")
            for col in range(8):
                state[row, col] = boardStr[row][col]

        return state

    def state2valueBoard(self):
        valueBoard = np.zeros(
            (14, 8, 8), dtype=np.uint8
        )  # 14 = 12 pieces + 2 (attack, threat)

        pieces = [
            "P",
            "N",
            "B",
            "R",
            "Q",
            "K",
            "p",
            "n",
            "b",
            "r",
            "q",
            "k",
            "ATTACK",
            "THREAT",
        ]

        PieceTables = {}
        for piece in pieces:
            PieceTables[piece] = np.zeros((8, 8), dtype=np.uint8)

        for row in range(8):
            for col in range(8):
                if self.state[row, col] != ".":
                    PieceTables[self.state[row, col]][row, col] = 1

        # Calculate attack and threat
        for row in range(8):
            for col in range(8):
                square = col * 8 + row
                # print(f"Square: {square} ({row}, {col})")

                # Calculate attack
                if self.board.is_attacked_by(chess.WHITE, square):
                    PieceTables["ATTACK"][col, row] = 1

                # Calculate threat
                if self.board.is_attacked_by(chess.BLACK, square):
                    PieceTables["THREAT"][col, row] = 1

                # PieceTables["ATTACK"] = np.rot90(PieceTables["ATTACK"], k=3)
                # PieceTables["THREAT"] = np.rot90(PieceTables["THREAT"], k=3)

        if self.board.turn == chess.WHITE:
            for piece in pieces:
                if piece == "ATTACK" or piece == "THREAT":
                    PieceTables[piece] = np.flipud(PieceTables[piece])

            # White pieces
            valueBoard[0] = PieceTables["P"]
            valueBoard[1] = PieceTables["N"]
            valueBoard[2] = PieceTables["B"]
            valueBoard[3] = PieceTables["R"]
            valueBoard[4] = PieceTables["Q"]
            valueBoard[5] = PieceTables["K"]

            # Black pieces
            valueBoard[6] = PieceTables["p"]
            valueBoard[7] = PieceTables["n"]
            valueBoard[8] = PieceTables["b"]
            valueBoard[9] = PieceTables["r"]
            valueBoard[10] = PieceTables["q"]
            valueBoard[11] = PieceTables["k"]

            # Attack
            valueBoard[12] = PieceTables["ATTACK"]

            # Threat
            valueBoard[13] = PieceTables["THREAT"]
        else:
            for piece in pieces:
                if piece != "ATTACK" and piece != "THREAT":
                    PieceTables[piece] = np.flipud(PieceTables[piece])

            # White pieces
            valueBoard[6] = PieceTables["P"]
            valueBoard[7] = PieceTables["N"]
            valueBoard[8] = PieceTables["B"]
            valueBoard[9] = PieceTables["R"]
            valueBoard[10] = PieceTables["Q"]
            valueBoard[11] = PieceTables["K"]

            # Black pieces
            valueBoard[0] = PieceTables["p"]
            valueBoard[1] = PieceTables["n"]
            valueBoard[2] = PieceTables["b"]
            valueBoard[3] = PieceTables["r"]
            valueBoard[4] = PieceTables["q"]
            valueBoard[5] = PieceTables["k"]

            # Attack
            valueBoard[13] = PieceTables["ATTACK"]

            # Threat
            valueBoard[12] = PieceTables["THREAT"]

        return valueBoard

    def __str__(self):
        string = f""
        string += f"***** Turn *****\n{'White' if self.board.turn else 'Black'}\n"
        string += f"***** Board *****\n{self.board}\n"
        string += f"***** State *****\n{self.state}\n"
        # string += f"***** Value Board *****\n{self.valueBoard}\n"
        string += f"-" * 50 + "\n"

        return string

    # show board
    def show_board(self, ax=None):
        if ax is None:
            if self.fig is None or self.ax is None:
                # print(f"Figure not initialized - call render() first")
                return

            self.show_board(self.ax)

            # Draw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            return

        # Clear figure
        ax.clear()

        # Set title
        ax.set_title(f"Chess Environment")

        # Remove axis
        ax.axis("off")

        # Create board
        self.boardImage = np.zeros(
            (self.resolution * 8, self.resolution * 8, 4), dtype=np.uint8
        )
        self.positionImage = np.zeros(
            (self.resolution * 8, self.resolution * 8, 4), dtype=np.uint8
        )

        # Draw colors
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    self.boardImage[
                        i * self.resolution : (i + 1) * self.resolution,
                        j * self.resolution : (j + 1) * self.resolution,
                    ] = self.boardColors[ColorType.WHITE]
                else:
                    self.boardImage[
                        i * self.resolution : (i + 1) * self.resolution,
                        j * self.resolution : (j + 1) * self.resolution,
                    ] = self.boardColors[ColorType.BLACK]

        # Draw pieces
        for i in range(8):
            for j in range(8):
                (pieceType, pieceColor) = PieceSymbols[self.state[i, j]]
                if pieceColor != ColorType.EMPTY:
                    self.positionImage[
                        i * self.resolution
                        + int(self.resolution * self.margin) : (i + 1) * self.resolution
                        - int(self.resolution * self.margin),
                        j * self.resolution
                        + int(self.resolution * self.margin) : (j + 1) * self.resolution
                        - int(self.resolution * self.margin),
                    ] = self.pieceImages[(pieceType, pieceColor)]

        # Show board markings at the center of each row on left and bottom
        for i in range(8):
            ax.text(
                -self.resolution / 4,
                i * self.resolution + self.resolution / 2,
                str(8 - i),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="black",
            )
            ax.text(
                i * self.resolution + self.resolution / 2,
                8.25 * self.resolution,
                chr(ord("A") + i),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                color="black",
            )

        # Show image
        ax.imshow(self.boardImage)
        ax.imshow(self.positionImage)

    def step(self, action):
        reward = 0
        done = False
        sound = None

        if not isinstance(action, str):
            if self.board.turn == chess.WHITE:
                action = action2moveMap[ColorType.WHITE][action]
            else:
                action = action2moveMap[ColorType.BLACK][action]

        print(f"Action: {action}")

        isValid = False
        move = None
        # Convert action to move
        try:
            move = chess.Move.from_uci(action)
            isValid = move in self.board.legal_moves
        except Exception as e:
            print(e)

        # Check if move is valid
        if isValid:
            reward += rewards[rewardType.VALID_MOVE]
            sound = rewardType.VALID_MOVE

            # Check if move is a capture
            opponent = chess.WHITE if self.board.turn == chess.BLACK else chess.BLACK
            if self.board.color_at(move.to_square) == opponent:
                reward += rewards[rewardType.CAPTURE]
                sound = rewardType.CAPTURE

            # Check if move is a check
            if self.board.gives_check(move):
                reward += rewards[rewardType.CHECK]
                sound = rewardType.CHECK

            # Check if move is a promotion
            if move.promotion is not None:
                reward += rewards[rewardType.PROMOTION]
                sound = rewardType.PROMOTION

            # Make move
            self.board.push(move)

            # Check if game is over
            if self.board.is_game_over():
                if self.board.is_checkmate():
                    reward += rewards[rewardType.WIN]
                    sound = rewardType.WIN
                elif (
                    self.board.is_stalemate()
                    or self.board.is_insufficient_material()
                    or self.board.is_fifty_moves()
                    or self.board.is_fivefold_repetition()
                ):
                    reward += rewards[rewardType.DRAW]
                    sound = rewardType.DRAW
                else:
                    reward += rewards[rewardType.LOSS]
                    sound = rewardType.LOSS
                done = True

        else:
            print("Invalid move")
            reward += rewards[rewardType.INVALID_MOVE]
            sound = rewardType.INVALID_MOVE
            done = True

        # Play sound
        if sound is not None and not self.mute:
            self.moveSounds[sound].play()

        # Update state
        self.state = self.board2state()
        self.valueBoard = self.state2valueBoard()
        self.count += 1

        return (
            self.valueBoard,
            reward,
            done,
            {
                "move": None if move is None else move.uci(),
                "move type": sound.name if sound is not None else None,
                "count": self.count,
                "turn": f"{'White' if self.board.turn == chess.WHITE else 'Black'}",
                "position": self.board.fen(),
                "legal_moves": [move.uci() for move in list(self.board.legal_moves)],
                "piece_val": np.sum(self.valueBoard),
                "check": self.board.is_check(),
                "checkmate": self.board.is_checkmate(),
                "stalemate": self.board.is_stalemate(),
                "insufficient_material": self.board.is_insufficient_material(),
                "fifty_moves": self.board.is_fifty_moves(),
                "fivefold_repetition": self.board.is_fivefold_repetition(),
            },
        )

    @staticmethod
    def squaretostr(square):
        return f"{chr(ord('a') + square[1])}{square[0]+1}"

    @staticmethod
    def strtosquare(square):
        return (int(square[1]) - 1, ord(square[0]) - ord("a"))


if __name__ == "__main__":
    env = ChessEnv()
    env.render()

    while True:
        env.reset()
        print(env)

        done = False
        while not done:
            # get click point from matplotlib
            clickPoint = plt.ginput(1, timeout=-1)[0]

            # convert click point to board coordinates
            clickPoint = (
                7 - int(clickPoint[1] / env.resolution),
                int(clickPoint[0] / env.resolution),
            )

            start = ChessEnv.squaretostr(clickPoint)
            print(f"Start: {start}")

            # get click point from matplotlib
            clickPoint = plt.ginput(1, timeout=-1)[0]

            # convert click point to board coordinates
            clickPoint = (
                7 - int(clickPoint[1] / env.resolution),
                int(clickPoint[0] / env.resolution),
            )

            end = ChessEnv.squaretostr(clickPoint)
            print(f"End: {end}")

            action = f"{start}{end}"

            # action = random.choice(list(env.board.legal_moves)).uci()
            print(action)

            obs, reward, done, info = env.step(action)

            print(env)

            env.render()
