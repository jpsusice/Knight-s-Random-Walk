import random
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

BOARD_SIZE             = 8
TRIALS_PER_TOUR_LENGTH = 5 # (results will be averaged)
MIN_TOUR_LENGTH        = 100
MAX_TOUR_LENGTH        = 500
DEGREE_OF_FIT          = 2

#Knight's moves
moves = numpy.array([(1, 2), (1, -2), (2, 1), (2, -1), (-1, 2), (-1, -2), (-2, 1), (-2, -1)])

class Board:
    class Piece:
        def __init__(self, x = 0, y = 0, moves = []):
            self.moves = moves
            self.x = x
            self.y = y
        def set_coordinates(self, x, y):
            self.x = x 
            self.y = y
        def get_coordinates(self):
            return (self.x, self.y)

    def generate_move_board(self, board_size, moves):
        move_board = {}
        for x in range(board_size):
            for y in range(board_size):
                a = []
                for move in self.piece.moves:
                    next_x = x + move[0] 
                    next_y = y + move[1]
                    if self.on_board(next_x, next_y):                 
                        a.append((next_x, next_y))
                move_board[(x, y)] = a
        return move_board

    def __init__(self, moves, x = 0, y = 0, board_size = 8):
        self.size  = board_size
        self.piece = self.Piece(x, y, moves)
        self.move_board = self.generate_move_board(self.size, moves)        


    def set_piece(self, x = 0, y = 0):
        self.piece.set_coordinates(x, y)

    def on_board(self, i, j):
        return 0 <= i < self.size and 0 <= j < self.size 

    def count_accessible(self, i, j):
        count = 0
        for move in self.piece.moves: 
            if self.on_board(i + move[0], j + move[1]):
                count += 1
        return count

    def random_move(self):
        (x, y) = self.piece.get_coordinates()
        (new_x, new_y) = random.choice(self.move_board[(x, y)])
        self.set_piece(new_x, new_y)
        return(new_x, new_y)
	 
    def theoretical_percentages(self):
        # Calculate the percentage of time the piece with move vector moves will 
        # stay at each square on the board in a random walk. Percentages are calculated 
        # using theory of stochastic processes
        board = numpy.zeros((self.size, self.size))
        count = 0
        for i in range(self.size):
            for j in range(self.size):
                num_accessible = self.count_accessible(i, j)
                board[i][j] = num_accessible
                count += num_accessible
        return board/count	

 


def run_tour(board, tour_length):
    counter_board = numpy.zeros((board.size, board.size), dtype = int)
    for t in range(tour_length):
        counter_board[board.piece.x][board.piece.y] += 1
        board.random_move()
    board.set_piece()
    return counter_board

def run_trials(board, tour_length, trials_per_tour_length):
    # returns avg. percent error per square
    deviations = numpy.array([])
    predicted_percentages = board.theoretical_percentages()
    total = 0
    for i in range(trials_per_tour_length):
        counter_board = run_tour(board, tour_length)
        experimental_percentages = counter_board/float(tour_length)
        avg_error_per_square = sum(sum(
                        abs(predicted_percentages - experimental_percentages)/predicted_percentages)) / board.size**2
        total += avg_error_per_square
    return total/float(trials_per_tour_length)   

def percent_errors(board, tour_lengths, trials_per_tour_length):
    errors = [run_trials(board, t, trials_per_tour_length) for t in tour_lengths]
    return errors


 
board = Board(moves, 0, 0, BOARD_SIZE)
tour_lengths = range(MIN_TOUR_LENGTH, MAX_TOUR_LENGTH + 1)
errors = percent_errors(board, tour_lengths, TRIALS_PER_TOUR_LENGTH)


coeffs = numpy.polyfit(tour_lengths, errors, DEGREE_OF_FIT)
regr_fcn = numpy.poly1d(coeffs)
plt.plot(tour_lengths, errors, tour_lengths, regr_fcn(tour_lengths))


plt.xlabel('Length of Knight\'s Tour')
plt.ylabel('Average Percent Error Per Square')
plt.title('Percent Error of Computational vs. Theoretical \n Square Occupancy Rates')
plt.show()



