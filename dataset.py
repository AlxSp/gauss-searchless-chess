
#%%
import os
import chess
import torch
import torch.nn as nn
from torch.special import erf
from torch.utils.data import Dataset, DataLoader
import numpy as np
from apache_beam import coders

from bagz import BagReader, BagDataSource

#%%
CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}

#%%
_CHESS_FILE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def _compute_all_possible_actions() -> tuple[dict[str, int], dict[int, str]]:
  """Returns two dicts converting moves to actions and actions to moves.

  These dicts contain all possible chess moves.
  """
  all_moves = []

  # First, deal with the normal moves.
  # Note that this includes castling, as it is just a rook or king move from one
  # square to another.
  board = chess.BaseBoard.empty()
  for square in range(64):
    next_squares = []

    # Place the queen and see where it attacks (we don't need to cover the case
    # for a bishop, rook, or pawn because the queen's moves includes all their
    # squares).
    board.set_piece_at(square, chess.Piece.from_symbol('Q'))
    next_squares += board.attacks(square)

    # Place knight and see where it attacks
    board.set_piece_at(square, chess.Piece.from_symbol('N'))
    next_squares += board.attacks(square)
    board.remove_piece_at(square)

    for next_square in next_squares:
      all_moves.append(
          chess.square_name(square) + chess.square_name(next_square)
      )

  # Then deal with promotions.
  # Only look at the last ranks.
  promotion_moves = []
  for rank, next_rank in [('2', '1'), ('7', '8')]:
    for index_file, file in enumerate(_CHESS_FILE):
      # Normal promotions.
      move = f'{file}{rank}{file}{next_rank}'
      promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]

      # Capture promotions.
      # Left side.
      if file > 'a':
        next_file = _CHESS_FILE[index_file - 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
      # Right side.
      if file < 'h':
        next_file = _CHESS_FILE[index_file + 1]
        move = f'{file}{rank}{next_file}{next_rank}'
        promotion_moves += [(move + piece) for piece in ['q', 'r', 'b', 'n']]
  all_moves += promotion_moves

  move_to_action, action_to_move = {}, {}
  for action, move in enumerate(all_moves):
    assert move not in move_to_action
    move_to_action[move] = action
    action_to_move[action] = move

  return move_to_action, action_to_move


MOVE_TO_ACTION, ACTION_TO_MOVE = _compute_all_possible_actions()
NUM_ACTIONS = len(MOVE_TO_ACTION)

#%%
# Tokenizer >>>

_CHARACTERS = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    'a',
    'b',
    'c',
    'd',
    'e',
    'f',
    'g',
    'h',
    'p',
    'n',
    'r',
    'k',
    'q',
    'P',
    'B',
    'N',
    'R',
    'Q',
    'K',
    'w',
    '.',
]
# pyfmt: enable
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 77

def tokenize(fen: str):
  """Returns an array of tokens from a fen string.

  We compute a tokenized representation of the board, from the FEN string.
  The final array of tokens is a mapping from this string to numbers, which
  are defined in the dictionary `_CHARACTERS_INDEX`.
  For the 'en passant' information, we convert the '-' (which means there is
  no en passant relevant square) to '..', to always have two characters, and
  a fixed length output.

  Args:
    fen: The board position in Forsyth-Edwards Notation.
  """
  # Extracting the relevant information from the FEN.
  board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
  board = board.replace('/', '')
  board = side + board

  indices = list()

  for char in board:
    if char in _SPACES_CHARACTERS:
      indices.extend(int(char) * [_CHARACTERS_INDEX['.']])
    else:
      indices.append(_CHARACTERS_INDEX[char])

  if castling == '-':
    indices.extend(4 * [_CHARACTERS_INDEX['.']])
  else:
    for char in castling:
      indices.append(_CHARACTERS_INDEX[char])
    # Padding castling to have exactly 4 characters.
    if len(castling) < 4:
      indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

  if en_passant == '-':
    indices.extend(2 * [_CHARACTERS_INDEX['.']])
  else:
    # En passant is a square like 'e3'.
    for char in en_passant:
      indices.append(_CHARACTERS_INDEX[char])

  # Three digits for halfmoves (since last capture) is enough since the game
  # ends at 50.
  halfmoves_last += '.' * (3 - len(halfmoves_last))
  indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

  # Three digits for full moves is enough (no game lasts longer than 999
  # moves).
  fullmoves += '.' * (3 - len(fullmoves))
  indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

  assert len(indices) == SEQUENCE_LENGTH

  return np.asarray(indices, dtype=np.uint8)

# <<< Tokenizer

#%%
def compute_return_buckets_from_returns(
    returns: np.ndarray,
    bins_edges: np.ndarray,
) -> np.ndarray:
  """Arranges the discounted returns into bins.

  The returns are put into the bins specified by `bin_edges`. The length of
  `bin_edges` is equal to the number of buckets minus 1. In case of a tie (if
  the return is exactly equal to an edge), we take the bucket right before the
  edge. See example below.
  This function is purely using np.searchsorted, so it's a good reference to
  look at.

  Examples:
  * bin_edges=[0.5] and returns=[0., 1.] gives the buckets [0, 1].
  * bin_edges=[-30., 30.] and returns=[-200., -30., 0., 1.] gives the buckets
    [0, 0, 1, 1].

  Args:
    returns: An array of discounted returns, rank 1.
    bins_edges: The boundary values of the return buckets, rank 1.

  Returns:
    An array of buckets, described as integers, rank 1.

  Raises:
    ValueError if `returns` or `bins_edges` are not of rank 1.
  """
  if len(returns.shape) != 1:
    raise ValueError(
        'The passed returns should be of rank 1. Got'
        f' rank={len(returns.shape)}.'
    )
  if len(bins_edges.shape) != 1:
    raise ValueError(
        'The passed bins_edges should be of rank 1. Got'
        f' rank{len(bins_edges.shape)}.'
    )
  return np.searchsorted(bins_edges, returns, side='left')

def _process_win_prob(
    win_prob: float,
    return_buckets_edges: np.ndarray,
) -> np.ndarray:
  return compute_return_buckets_from_returns(
      returns=np.asarray([win_prob]),
      bins_edges=return_buckets_edges,
  )

def get_uniform_buckets_edges_values(
    num_buckets: int,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns edges and values of uniformly sampled buckets in [0, 1].

  Example: for num_buckets=4, it returns:
  edges=[0.25, 0.50, 0.75]
  values=[0.125, 0.375, 0.625, 0.875]

  Args:
    num_buckets: Number of buckets to create.
  """
  full_linspace = np.linspace(0.0, 1.0, num_buckets + 1)
  edges = full_linspace[1:-1]
  values = (full_linspace[:-1] + full_linspace[1:]) / 2
  return edges, values

#%%
class ScalarsToHLGauss(nn.Module):
  def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
    super().__init__()
    self.min_value = min_value
    self.max_value = max_value
    self.num_bins = num_bins
    self.sigma = sigma
    self.support = torch.linspace(
    min_value, max_value, num_bins + 1, dtype=torch.float32
    )

  def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
    cdf_evals = erf(
    (self.support - target.unsqueeze(-1))
    / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
    )
    z = cdf_evals[..., -1] - cdf_evals[..., 0]
    bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return bin_probs / z.unsqueeze(-1)


#%%
CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}

#%% action value decoder
action_value_decoder =  coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
))

#%%
class ActionValueDataset(Dataset):
    """
    This dataset class reads action_value.bagz files and converts the data into a sequence of state, action, and return bucket.
    The output is a sequence of the 77 states plus the action and return bucket, and the loss mask which attends to the state and action but not the return bucket.
    """

    def __init__(self, file_paths, hl_gauss=False):
        self.file_paths = file_paths
        self.num_return_buckets = 128
        self.hl_gauss = ScalarsToHLGauss(0.0, float(self.num_return_buckets), self.num_return_buckets, 0.96) if hl_gauss else None

        self.lengths = []
        for file_path in self.file_paths:
            self.lengths.append(len(BagReader(file_path)))

        self.length = sum(self.lengths)

        self.sample_sequence_length = SEQUENCE_LENGTH + 1 # (s) + (a) + (r)

        self._return_buckets_edges, _ = get_uniform_buckets_edges_values(
            self.num_return_buckets,
        )

        # The loss mask ensures that we only train on the error of the return bucket.
        self._loss_mask = np.full(
                shape=(self.sample_sequence_length,),
                fill_value=True,
                dtype=bool,
            )

        self._loss_mask[-1] = False

    def _get_record_index(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                return i, idx
            idx -= length

    def __len__(self):
        return self.length
    
    def _convert(self, sample):
        fen, move, win_prob = action_value_decoder.decode(sample)

        state = tokenize(fen).astype(np.int32)
        action = np.asarray([MOVE_TO_ACTION[move]], dtype=np.int32)
        return_bucket = _process_win_prob(win_prob, self._return_buckets_edges)[0]
        
        if self.hl_gauss:
           return_bucket = self.hl_gauss.transform_to_probs(torch.tensor(return_bucket))

        sequence = np.concatenate([state, action])

        assert len(sequence) == self.sample_sequence_length
        #assert len(self._loss_mask) == self.sample_sequence_length

        return sequence, return_bucket

    def __getitem__(self, idx):
        #TODO: add out of bounds check
        file_idx, record_idx = self._get_record_index(idx)
        return self._convert(BagReader(self.file_paths[file_idx])[record_idx])
    
#%%
if __name__ == '__main__':
    from os import listdir
    from os.path import isfile, join

    train_dir = os.path.join("data", "train")

    train_files = [os.path.join(train_dir, f) for f in listdir(train_dir) if isfile(join(train_dir, f))  and f.startswith("action_value")]

    #%%
    b = BagReader(train_files[0])

    len(b)


    #%%
    ds = ActionValueDataset(train_files, hl_gauss=True)
    len(ds)

    #%%
    ds._get_record_index(11177425)

    #%%
    len(ds)


    ds[16374]