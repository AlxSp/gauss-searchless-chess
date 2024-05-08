#%%
import chess
from apache_beam import coders

import numpy as np

from bagz import BagReader, BagDataSource

#%%
# This file runs the code of the original ConvertActionValueDataToSequence class

#%%
file_path = "/ubuntu_data/searchless_chess/data/test/action_value_data.bag"
file_path = "/ubuntu_data/searchless_chess/data/train/action_value-00000-of-02148_data.bag"
# %%
#%%
bagd = BagDataSource(file_path)

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

#%%
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

#%%
bagr = BagReader(file_path)

CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
fen, move, win_prob =  coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
    CODERS['win_prob'],
)).decode(bagr[0])

fen, move, win_prob
# fen gets converted to state
# move gets converted to action
# win_prob gets converted to return_bucket

#%%
state = tokenize(fen).astype(np.int32)
state
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

action = np.asarray([MOVE_TO_ACTION[move]], dtype=np.int32)

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



_sequence_length = SEQUENCE_LENGTH + 2  # (s) + (a) + (r)
num_return_buckets = 128

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

_return_buckets_edges, _ = get_uniform_buckets_edges_values(
        num_return_buckets,
    )
    # The loss mask ensures that we only train on the return bucket.

_loss_mask = np.full(
        shape=(_sequence_length,),
        fill_value=True,
        dtype=bool,
    )

_loss_mask[-1] = False


#%%
return_bucket = _process_win_prob(win_prob, _return_buckets_edges)

return_bucket


#%%
sequence = np.concatenate([state, action, return_bucket])

sequence, _loss_mask

assert len(sequence) == _sequence_length
assert len(_loss_mask) == _sequence_length


#%%
# BEHAVIORAL CLONING


file_path = "/ubuntu_data/searchless_chess/data/test/behavioral_cloning_data.bag"

bagr = BagReader(file_path)

CODERS = {
    'fen': coders.StrUtf8Coder(),
    'move': coders.StrUtf8Coder(),
    'count': coders.BigIntegerCoder(),
    'win_prob': coders.FloatCoder(),
}
fen, move =  coders.TupleCoder((
    CODERS['fen'],
    CODERS['move'],
)).decode(bagr[0])

#%%
state = tokenize(fen).astype(np.int32)
state

#%%
action = np.asarray([MOVE_TO_ACTION[move]], dtype=np.int32)

#%%
sequence = np.concatenate([state, action])

_sequence_length = SEQUENCE_LENGTH + 1  # (s) + (a)

_loss_mask = np.full(
        shape=(_sequence_length,),
        fill_value=True,
        dtype=bool,
    )
_loss_mask[-1] = False

assert len(sequence) == _sequence_length
assert len(_loss_mask) == _sequence_length


#%%
file_path = "/ubuntu_data/searchless_chess/data/test/state_value_data.bag"

bagr = BagReader(file_path)

fen, win_prob = coders.TupleCoder((
    CODERS['fen'],
    CODERS['win_prob'],
)).decode(bagr[0])

#%%
state = tokenize(fen).astype(np.int32)
state

#%%
return_bucket = _process_win_prob(win_prob, _return_buckets_edges)
return_bucket

#%%
sequence = np.concatenate([state, return_bucket])

_sequence_length = SEQUENCE_LENGTH + 1  # (s) + (r)

_loss_mask = np.full(
        shape=(_sequence_length,),
        fill_value=True,
        dtype=bool,
    )
_loss_mask[-1] = False 

assert len(sequence) == _sequence_length
assert len(_loss_mask) == _sequence_length
