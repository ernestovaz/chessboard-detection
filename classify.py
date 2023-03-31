import sys
import os
import cv2
import math
import numpy as np

from preprocess import zero_pad_image, get_edges
import chessboard

GT_DIR = 'ground_truth/green'
MAX = 10000000

SYMBOLS = {
    'pawn':     'p',
    'rook':     'r',
    'knight':   'n',
    'bishop':   'b',
    'queen':    'q',
    'king':     'k'
}

def sum_squared_diff(img1, img2):
    sum = 0
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            sum+= math.sqrt(pow(float(img1[y][x]) - float(img2[y][x]),2))

    return sum/(img1.shape[0]*img1.shape[1])

def calc_dist(img1, img2):
    img1 = zero_pad_image(img1, img2.shape)
    return sum_squared_diff(img1, img2)

def classify(img):
    min_dist = MAX
    label = "pawn"
    #f9f9f9
    #525457
    unique, colors = np.unique(
        img.reshape(-1, img.shape[-1]),
        axis=0,
        return_counts=True
    )
    if len(colors) < 20:
        return None, None

    white = np.where(np.all(unique == [249, 249, 249], axis=-1))[0]
    black = np.where(np.all(unique == [82, 84, 87], axis=-1))[0]
    if len(white) == 0:
        color = 'black'
    elif len(black) == 0 or colors[white] > colors[black]:
        color = 'white'
    else:
        color = 'black'

    for piece in chessboard.POSITIONS:
        edge_map = get_edges(img)
        avg_img_pth = f'{GT_DIR}/{piece}/avg.png'
        if os.path.exists(avg_img_pth):
            avg_img = cv2.imread(avg_img_pth, cv2.IMREAD_GRAYSCALE)
            dist = calc_dist(edge_map, avg_img)
            if dist < min_dist:
                min_dist = dist
                label = piece
    return label, color

def main():
    filename = sys.argv[1]
    input_image = cv2.imread(filename)
    board = chessboard.from_image(input_image)
    for y in range(7, -1, -1):
        consecutive_empties = 0
        for x in range(8):
            label, color = classify(board[x][y])
            if label is None:
                consecutive_empties += 1
            else:
                if consecutive_empties != 0:
                    print(consecutive_empties, end='')
                    consecutive_empties = 0
                symbol = SYMBOLS[label]
                if color == 'white':
                    symbol = symbol.upper()
                print(symbol, end='')
        if consecutive_empties != 0:
            print(consecutive_empties, end='')
            consecutive_empties = 0
        print('\\', end='')
    print('')

if __name__ == '__main__':
    main()
