import numpy as np
import cv2
import sys
import os


def get_output_filename():
    count = 0
    while os.path.exists('output_%d.png' % count):
        count += 1
    return 'output_%d.png' % count


def read_chessboard(image, corner_array):
    chessboard = np.empty((8,8), dtype='object')
    corners = np.array(corner_array).reshape(7, 7, 2)
    corners = np.swapaxes(corners, 0, 1)

    width, height, channels = image.shape
    for y in range(8):
        for x in range(8):
            top_left = np.copy(corners[max(0, x-1)][max(0, y-1)])+1
            if x == 0: top_left[0] = 0
            if y == 0: top_left[1] = 0

            bot_right = np.copy(corners[min(6, x)][min(6, y)])
            if x == 7: bot_right[0] = width-1
            if y == 7: bot_right[1] = height-1

            chessboard[x][y] = image[
                round(top_left[1]):round(bot_right[1]),
                round(top_left[0]):round(bot_right[0])
            ]

    return chessboard

def main():
    filename = sys.argv[1]
    img = cv2.imread(filename)

    found, corners = cv2.findChessboardCorners(img, (7,7))
    if found:
        chessboard = read_chessboard(img, corners)
        for y in range(8):
            for x in range(8):
                piece = chessboard[x][y]
                edges = cv2.Canny(piece, 100, 200)
                if edges.mean() > 0.01:
                    cv2.imwrite(get_output_filename(), edges)

if __name__ == '__main__':
    main()



