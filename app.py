
from flask import Flask, render_template, session, redirect, url_for, request
import os
import random
import requests
from dotenv import load_dotenv
from error import register_error_handlers

load_dotenv()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Add a secret key for sessions
register_error_handlers(app)

# Chess classes
class Piece:
    def __init__(self, color, symbol):
        self.color = color  # 'white' or 'black'
        self.symbol = symbol

    def get_possible_moves(self, board, row, col):
        # To be overridden by subclasses
        return []

    def to_string(self):
        return ('w' if self.color == 'white' else 'b') + self.symbol

class Pawn(Piece):
    def __init__(self, color):
        super().__init__(color, 'P')

    def get_possible_moves(self, board, row, col):
        moves = []
        direction = -1 if self.color == 'white' else 1
        # Move forward
        if 0 <= row + direction < 8 and board[row + direction][col] is None:
            moves.append((row + direction, col))
            # Double move from starting position
            if (self.color == 'white' and row == 6) or (self.color == 'black' and row == 1):
                if board[row + 2 * direction][col] is None:
                    moves.append((row + 2 * direction, col))
        # Captures
        for dc in [-1, 1]:
            if 0 <= col + dc < 8 and 0 <= row + direction < 8:
                target = board[row + direction][col + dc]
                if target and target.color != self.color:
                    moves.append((row + direction, col + dc))
        return moves

class Rook(Piece):
    def __init__(self, color):
        super().__init__(color, 'R')

    def get_possible_moves(self, board, row, col):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                if board[r][c] is None:
                    moves.append((r, c))
                elif board[r][c].color != self.color:
                    moves.append((r, c))
                    break
                else:
                    break
                r += dr
                c += dc
        return moves

class Knight(Piece):
    def __init__(self, color):
        super().__init__(color, 'N')

    def get_possible_moves(self, board, row, col):
        moves = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in deltas:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                if board[r][c] is None or board[r][c].color != self.color:
                    moves.append((r, c))
        return moves

class Bishop(Piece):
    def __init__(self, color):
        super().__init__(color, 'B')

    def get_possible_moves(self, board, row, col):
        moves = []
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                if board[r][c] is None:
                    moves.append((r, c))
                elif board[r][c].color != self.color:
                    moves.append((r, c))
                    break
                else:
                    break
                r += dr
                c += dc
        return moves

class Queen(Piece):
    def __init__(self, color):
        super().__init__(color, 'Q')

    def get_possible_moves(self, board, row, col):
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                if board[r][c] is None:
                    moves.append((r, c))
                elif board[r][c].color != self.color:
                    moves.append((r, c))
                    break
                else:
                    break
                r += dr
                c += dc
        return moves

class King(Piece):
    def __init__(self, color):
        super().__init__(color, 'K')

    def get_possible_moves(self, board, row, col):
        moves = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    if board[r][c] is None or board[r][c].color != self.color:
                        moves.append((r, c))
        return moves

def piece_from_string(s):
    if not s:
        return None
    color = 'white' if s[0] == 'w' else 'black'
    symbol = s[1]
    if symbol == 'P':
        return Pawn(color)
    elif symbol == 'R':
        return Rook(color)
    elif symbol == 'N':
        return Knight(color)
    elif symbol == 'B':
        return Bishop(color)
    elif symbol == 'Q':
        return Queen(color)
    elif symbol == 'K':
        return King(color)
    return None

def board_to_objects(board_strings):
    return [[piece_from_string(cell) for cell in row] for row in board_strings]

def is_check(board, color):
    # Find king position
    king_pos = None
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece and piece.symbol == 'K' and piece.color == color:
                king_pos = (i, j)
                break
        if king_pos:
            break
    if not king_pos:
        return False  # No king? Weird
    # Check if any opponent piece can attack king
    opponent_color = 'black' if color == 'white' else 'white'
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece and piece.color == opponent_color:
                moves = piece.get_possible_moves(board, i, j)
                if king_pos in moves:
                    return True
    return False

def is_checkmate(board, color):
    if not is_check(board, color):
        return False
    # Check if any move can get out of check
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece and piece.color == color:
                moves = piece.get_possible_moves(board, i, j)
                for move in moves:
                    # Simulate move
                    temp_board = [row[:] for row in board]
                    temp_board[move[0]][move[1]] = temp_board[i][j]
                    temp_board[i][j] = None
                    if not is_check(temp_board, color):
                        return False
    return True

def initialize_chess_board():
    board = [['' for _ in range(8)] for _ in range(8)]
    # Place pieces
    # White
    board[7][0] = Rook('white').to_string()
    board[7][1] = Knight('white').to_string()
    board[7][2] = Bishop('white').to_string()
    board[7][3] = Queen('white').to_string()
    board[7][4] = King('white').to_string()
    board[7][5] = Bishop('white').to_string()
    board[7][6] = Knight('white').to_string()
    board[7][7] = Rook('white').to_string()
    for i in range(8):
        board[6][i] = Pawn('white').to_string()
    # Black
    board[0][0] = Rook('black').to_string()
    board[0][1] = Knight('black').to_string()
    board[0][2] = Bishop('black').to_string()
    board[0][3] = Queen('black').to_string()
    board[0][4] = King('black').to_string()
    board[0][5] = Bishop('black').to_string()
    board[0][6] = Knight('black').to_string()
    board[0][7] = Rook('black').to_string()
    for i in range(8):
        board[1][i] = Pawn('black').to_string()
    return board

@app.route('/')
def index():
    if 'game' not in session:
        return render_template('mode.html')
    if 'mode' not in session:
        return render_template('mode.html')
    if session['game'] == 'tictactoe':
        if 'board' not in session:
            session['board'] = [['' for _ in range(3)] for _ in range(3)]
            session['turn'] = 'X'
            session['winner'] = None
        return render_template('index.html')
    elif session['game'] == 'chess':
        if 'board' not in session:
            session['board'] = initialize_chess_board()
            session['turn'] = 'white'
            session['winner'] = None
            session['captured_white'] = []
            session['captured_black'] = []
        return render_template('chess.html')
    return render_template('mode.html')

@app.route('/game/<game>')
def select_game(game):
    if game in ['tictactoe', 'chess']:
        session['game'] = game
    return redirect(url_for('index'))

@app.route('/mode/<mode>')
def set_mode(mode):
    if mode in ['human', 'computer']:
        session['mode'] = mode
    return redirect(url_for('index'))

@app.route('/move/<int:row>/<int:col>')
def move(row, col):
    if 'mode' not in session:
        return redirect(url_for('index'))
    if session['winner'] or session['board'][row][col]:
        return redirect(url_for('index'))
    session['board'][row][col] = session['turn']
    if check_winner(session['board'], session['turn']):
        session['winner'] = session['turn']
    elif is_full(session['board']):
        session['winner'] = 'Draw'
    else:
        session['turn'] = 'O' if session['turn'] == 'X' else 'X'
        if session.get('mode') == 'computer' and session['turn'] == 'O' and not session['winner']:
            computer_move()
    return redirect(url_for('index'))

@app.route('/select_piece/<int:row>/<int:col>')
def select_piece(row, col):
    if 'game' not in session or session['game'] != 'chess':
        return redirect(url_for('index'))
    if session['winner']:
        return redirect(url_for('index'))
    board = board_to_objects(session['board'])
    piece = board[row][col]
    if not piece or piece.color != session['turn']:
        session.pop('selected', None)
        session.pop('possible_moves', None)
        return redirect(url_for('index'))
    session['selected'] = [row, col]
    moves = piece.get_possible_moves(board, row, col)
    session['possible_moves'] = moves
    return redirect(url_for('index'))

@app.route('/move_to/<int:row>/<int:col>')
def move_to(row, col):
    if 'game' not in session or session['game'] != 'chess':
        return redirect(url_for('index'))
    if session['winner']:
        return redirect(url_for('index'))
    from_row = None
    from_col = None
    if 'from' in request.args:
        parts = request.args['from'].split(',')
        from_row = int(parts[0])
        from_col = int(parts[1])
    elif 'selected' in session:
        from_row, from_col = session['selected']
    else:
        return redirect(url_for('index'))
    board = board_to_objects(session['board'])
    piece = board[from_row][from_col]
    if not piece or piece.color != session['turn']:
        return redirect(url_for('index'))
    moves = piece.get_possible_moves(board, from_row, from_col)
    if (row, col) not in moves:
        return redirect(url_for('index'))
    # Make move
    if session['board'][row][col]:
        captured = session['board'][row][col]
        if captured[0] == 'w':
            session['captured_white'].append(captured)
        else:
            session['captured_black'].append(captured)
    session['board'][row][col] = session['board'][from_row][from_col]
    session['board'][from_row][from_col] = ''
    session.pop('selected', None)
    session.pop('possible_moves', None)
    # Switch turn
    session['turn'] = 'black' if session['turn'] == 'white' else 'white'
    # Check for checkmate
    board = board_to_objects(session['board'])
    if is_checkmate(board, session['turn']):
        session['winner'] = 'white' if session['turn'] == 'black' else 'black'
    elif is_check(board, session['turn']):
        # Maybe set a message, but for now just continue
        pass
    if session.get('mode') == 'computer' and session['turn'] == 'black' and not session['winner']:
        computer_move_chess()
    return redirect(url_for('index'))

@app.route('/make_chess_move', methods=['POST'])
def make_chess_move():
    if 'game' not in session or session['game'] != 'chess':
        return redirect(url_for('index'))
    if session['winner']:
        return redirect(url_for('index'))
    from_row = int(request.form['from_row'])
    from_col = int(request.form['from_col'])
    to_row = int(request.form['to_row'])
    to_col = int(request.form['to_col'])
    board = board_to_objects(session['board'])
    piece = board[from_row][from_col]
    if not piece or piece.color != session['turn']:
        return redirect(url_for('index'))
    moves = piece.get_possible_moves(board, from_row, from_col)
    if (to_row, to_col) not in moves:
        return redirect(url_for('index'))
    # Make move
    session['board'][to_row][to_col] = session['board'][from_row][from_col]
    session['board'][from_row][from_col] = ''
    # Switch turn
    session['turn'] = 'black' if session['turn'] == 'white' else 'white'
    # Check for checkmate
    board = board_to_objects(session['board'])
    if is_checkmate(board, session['turn']):
        session['winner'] = 'white' if session['turn'] == 'black' else 'black'
    elif is_check(board, session['turn']):
        # Maybe set a message, but for now just continue
        pass
    if session.get('mode') == 'computer' and session['turn'] == 'black' and not session['winner']:
        computer_move_chess()
    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    session.pop('board', None)
    session.pop('turn', None)
    session.pop('winner', None)
    session.pop('game', None)
    session.pop('mode', None)
    session.pop('selected', None)
    session.pop('possible_moves', None)
    session.pop('captured_white', None)
    session.pop('captured_black', None)
    return redirect(url_for('index'))

def check_winner(board, player):
    # Check rows, columns, diagonals
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2-i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return all(all(cell for cell in row) for row in board)

def minimax(board, depth, is_maximizing):
    if check_winner(board, 'O'):
        return 10 - depth
    if check_winner(board, 'X'):
        return depth - 10
    if is_full(board):
        return 0

    if is_maximizing:
        max_eval = -float('inf')
        for i in range(3):
            for j in range(3):
                if not board[i][j]:
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, False)
                    board[i][j] = ''
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if not board[i][j]:
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, True)
                    board[i][j] = ''
                    min_eval = min(min_eval, eval)
        return min_eval

def computer_move():
    move = get_ai_move_tictactoe(session['board'])
    if move:
        session['board'][move[0]][move[1]] = 'O'
    else:
        # Fallback to minimax
        board = session['board']
        best_score = -float('inf')
        best_move = None
        for i in range(3):
            for j in range(3):
                if not board[i][j]:
                    board[i][j] = 'O'
                    score = minimax(board, 0, False)
                    board[i][j] = ''
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
        if best_move:
            session['board'][best_move[0]][best_move[1]] = 'O'
    if check_winner(session['board'], 'O'):
        session['winner'] = 'O'
    elif is_full(session['board']):
        session['winner'] = 'Draw'
    else:
        session['turn'] = 'X'

def piece_value(piece):
    if not piece:
        return 0
    values = {'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000}
    return values.get(piece.symbol, 0)

# Position tables (for white, flip for black)
pawn_table = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
]

knight_table = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
]

bishop_table = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
]

rook_table = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [5, 10, 10, 10, 10, 10, 10,  5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [-5,  0,  0,  0,  0,  0,  0, -5],
    [0,  0,  0,  5,  5,  0,  0,  0]
]

queen_table = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [-5,  0,  5,  5,  5,  5,  0, -5],
    [0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20]
]

king_table = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [20, 20,  0,  0,  0,  0, 20, 20],
    [20, 30, 10,  0,  0, 10, 30, 20]
]

def get_ai_move_tictactoe(board):
    prompt = "You are playing tic tac toe as O. The board is:\n"
    for row in board:
        prompt += ' '.join(cell if cell else '.' for cell in row) + '\n'
    prompt += "Make the best move. Respond with row,col (0-2)"
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        if response.status_code == 200:
            move = response.json()['choices'][0]['message']['content'].strip()
            r, c = map(int, move.split(','))
            if 0 <= r < 3 and 0 <= c < 3 and not board[r][c]:
                return r, c
    except:
        pass
    return None

def get_ai_move_chess(board, turn):
    prompt = f"You are playing chess as {turn}. The board is:\n"
    for i in range(8):
        row = []
        for j in range(8):
            piece = board[i][j]
            row.append(piece if piece else '.')
        prompt += ' '.join(row) + '\n'
    prompt += "Make the best move. Respond with from_row,from_col,to_row,to_col (0-7)"
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=10
        )
        if response.status_code == 200:
            move = response.json()['choices'][0]['message']['content'].strip()
            fr, fc, tr, tc = map(int, move.split(','))
            if all(0 <= x < 8 for x in [fr, fc, tr, tc]):
                return fr, fc, tr, tc
    except:
        pass
    return None

def position_value(piece, row, col):
    if not piece:
        return 0
    table = None
    if piece.symbol == 'P':
        table = pawn_table
    elif piece.symbol == 'N':
        table = knight_table
    elif piece.symbol == 'B':
        table = bishop_table
    elif piece.symbol == 'R':
        table = rook_table
    elif piece.symbol == 'Q':
        table = queen_table
    elif piece.symbol == 'K':
        table = king_table
    if table:
        if piece.color == 'white':
            return table[row][col]
        else:
            return table[7-row][col]  # Flip for black
    return 0

def evaluate_board(board):
    score = 0
    for i in range(8):
        for j in range(8):
            piece = board[i][j]
            if piece:
                value = piece_value(piece) + position_value(piece, i, j)
                if piece.color == 'black':
                    score += value
                else:
                    score -= value
    # Penalize if king in check
    if is_check(board, 'white'):
        score += 50  # Bonus for putting white in check
    if is_check(board, 'black'):
        score -= 50  # Penalty for being in check
    return score

def computer_move_chess():
    move = get_ai_move_chess(session['board'], 'black')
    if move:
        from_row, from_col, to_row, to_col = move
        session['board'][to_row][to_col] = session['board'][from_row][from_col]
        session['board'][from_row][from_col] = ''
    else:
        # Fallback to evaluation
        board = board_to_objects(session['board'])
        best_score = -float('inf')
        best_move = None
        for i in range(8):
            for j in range(8):
                piece = board[i][j]
                if piece and piece.color == 'black':
                    moves = piece.get_possible_moves(board, i, j)
                    for mv in moves:
                        temp_board = [r[:] for r in board]
                        temp_board[mv[0]][mv[1]] = temp_board[i][j]
                        temp_board[i][j] = None
                        score = evaluate_board(temp_board)
                        if score > best_score:
                            best_score = score
                            best_move = (i, j, mv[0], mv[1])
        if best_move:
            from_row, from_col, to_row, to_col = best_move
            session['board'][to_row][to_col] = session['board'][from_row][from_col]
            session['board'][from_row][from_col] = ''
    session['turn'] = 'white'
    board = board_to_objects(session['board'])
    if is_checkmate(board, 'white'):
        session['winner'] = 'black'
    session.pop('selected', None)
    session.pop('possible_moves', None)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
