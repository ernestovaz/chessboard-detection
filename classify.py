import sys
import os
import cv2
import math

from preprocess import zero_pad_image, get_edges
import chessboard

GT_DIR = 'ground_truth/green'
MAX = 10000000

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
    for piece in chessboard.POSITIONS:
        avg_img_pth = f'{GT_DIR}/{piece}/avg.png'
        if os.path.exists(avg_img_pth):
            avg_img = cv2.imread(avg_img_pth, cv2.IMREAD_GRAYSCALE)
            dist = calc_dist(img, avg_img)
            if dist < min_dist:
                min_dist = dist
                label = piece
    return label

def main():
    filename = sys.argv[1]
    input_image = cv2.imread(filename)
    board = chessboard.from_image(input_image)
    for y in range(7, -1, -1):
        for x in range(8):
            edge_map = get_edges(board[x][y])
            label = classify(edge_map)
            print(label + ' ', end='')
        print('')

if __name__ == '__main__':
    main()
