import numpy as np


class GameBoard:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.state = np.zeros((rows, cols), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.is_game_over = False
        self.winner = None

    def _set_state_and_player(self, state, current_player):
        return (
            state if state is not None else self.state,
            current_player if current_player is not None else self.current_player,
        )

    def reset(self):
        self.state.fill(0)
        self.current_player = 1
        self.is_game_over = False
        self.winner = None

    def make_move(self, col):
        """
        Update the current game state with a move. Assumes
        that the given move has been checked for validity already

        Returns:
            bool: game over flag
        """
        row = self._find_empty_row(col, self.state)
        self.state[row, col] = self.current_player

        if self._check_win(row, col, self.state, self.current_player):
            self.is_game_over = True
            self.winner = self.current_player
        elif np.all(self.state[0, :] != 0):
            self.is_game_over = True
            self.winner = None  # Draw

        self.current_player = 3 - self.current_player  # Switch player between 1 and 2

    def next_state(self, col, state, current_player):
        """
        Find the next state without updating internal state. Used during MCTS search.

        Returns:
            bool: game over flag
        """
        row = self._find_empty_row(col, state)
        state[row, col] = current_player

        return state, self._check_win(row, col, state, current_player)

    def is_valid_move(self, col, state=None):
        state, _ = self._set_state_and_player(state, None)
        return state[0, col] == 0

    def _find_empty_row(self, col, state):
        for row in reversed(range(self.rows)):
            if state[row, col] == 0:
                return row

    def _check_win(self, row, col, state, current_player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self._check_direction(row, col, dr, dc, state, current_player):
                return True
        return False

    def _check_direction(self, row, col, dr, dc, state, current_player):
        count = 1  # Count the original piece
        for i in range(1, 4):  # Check three additional pieces
            r, c = row + dr * i, col + dc * i
            if (
                0 <= r < self.rows
                and 0 <= c < self.cols
                and state[r, c] == current_player
            ):
                count += 1
            else:
                break
        for i in range(1, 4):  # Check the opposite direction
            r, c = row - dr * i, col - dc * i
            if (
                0 <= r < self.rows
                and 0 <= c < self.cols
                and state[r, c] == current_player
            ):
                count += 1
            else:
                break
        return count >= 4

    def display_board(self):
        print(self.state)

    @staticmethod
    def generate_legal_moves(state):
        """
        Generate a list of legal moves based on a given board state.

        Args:
        - state (numpy.ndarray): The board state to evaluate.

        Returns:
        - list: A list of integers representing the columns where a move can be made.
        """
        return [col for col in range(state.shape[1]) if state[0, col] == 0]

    def encode_board(self, board=None, current_player=None):
        """
        Encode an arbitrary board state. If no board is given, encode the current state.

        Args:
        - board (numpy.ndarray): The game board as a 2D NumPy array of shape (6, 7).
        - current_player (int): The current player (1 or 2).

        Returns:
        - numpy.ndarray: The encoded board as a 3D array of shape (3, 6, 7).
        """
        if board is None and current_player is None:
            board = self.state
            current_player = self.current_player

        encoded_board = np.zeros((3, 6, 7), dtype=np.float32)
        encoded_board[0] = board == 1
        encoded_board[1] = board == 2
        encoded_board[2] = 1 if current_player == 2 else 0

        return encoded_board

    def decode_state(self, encoded_board):
        """
        Decode a board state.

        Args:
        - encoded_board (numpy.ndarray): The encoded board as a 3D array of shape (3, 6, 7).

        Returns:
        - board (numpy.ndarray): The game board as a 2D NumPy array of shape (6, 7).
        - current_player (int): The current player (1 or 2).
        """

        board = np.zeros((6, 7), dtype=int)
        board[encoded_board[0] == 1] = 1
        board[encoded_board[1] == 1] = 2

        current_player = 2 if np.all(encoded_board[2] == 1) else 1

        return board, current_player


class GameBoard2:
    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.state = np.zeros((rows, cols), dtype=int)
        self.current_player = 1  # Player 1 starts
        self.is_game_over = False
        self.winner = None

    def reset(self):
        self.state.fill(0)
        self.current_player = 1
        self.is_game_over = False
        self.winner = None

    def make_move(self, col):
        """
        Update the current game state with a move. Assumes
        that the given move has been checked for validity already

        Returns:
            bool: game over flag
        """
        row = self._find_empty_row(col)
        self.state[row, col] = self.current_player

        if self.check_winner(row, col):
            self.is_game_over = True
            self.winner = self.current_player
        self.current_player = 3 - self.current_player  # Switch player between 1 and 2

    def is_valid_move(self, col):
        return self.state[0, col] == 0

    def _find_empty_row(self, col):
        for row in reversed(range(self.rows)):
            if self.state[row, col] == 0:
                return row

    def check_winner(self, row, col):
        # Horizontal, vertical, and diagonal checks
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self._check_direction(row, col, dr, dc):
                return True
        return False

    def _check_direction(self, row, col, dr, dc):
        count = 1  # Count the original piece
        for i in range(1, 4):  # Check three additional pieces
            r, c = row + dr * i, col + dc * i
            if (
                0 <= r < self.rows
                and 0 <= c < self.cols
                and self.state[r, c] == self.current_player
            ):
                count += 1
            else:
                break
        for i in range(1, 4):  # Check the opposite direction
            r, c = row - dr * i, col - dc * i
            if (
                0 <= r < self.rows
                and 0 <= c < self.cols
                and self.state[r, c] == self.current_player
            ):
                count += 1
            else:
                break
        return count >= 4

    def display_board(self):
        print(self.state)  # Flip the board so that the bottom row appears at the bottom

    ###### MCTS Helpers ######

    @staticmethod
    def next_state(self, current_state, player, col):
        row = self._find_empty_row(col)
        self.state[row, col] = self.current_player

        if self.check_winner(row, col):
            self.is_game_over = True
            self.winner = self.current_player
        self.current_player = 3 - self.current_player  # Switch player between 1 and 2
