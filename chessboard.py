import numpy as np
import cv2

# Thought about creating an Enum for pieces 
# settled on using strings for simplicity
#
POSITIONS = {
    'pawn':     [(x,y) for x in range(8) for y in [1,6]],
    'rook':     [(0,0), (7,0), (0,7), (7,7)],
    'knight':   [(1,0), (6,0), (1,7), (6,7)],
    'bishop':   [(2,0), (5,0), (2,7), (5,7)],
    'queen':    [(3,0), (3,7)],
    'king':     [(4,0), (4,7)]
}


# Constructs chessboard matrix containing piece images,
# accessed with indices 0 through 7, origin in bottom-left corner
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

            chessboard[x][7 - y] = image[
                round(top_left[1]):round(bot_right[1]),
                round(top_left[0]):round(bot_right[0])
            ]

    return chessboard


def from_image(image):
    found, corners = cv2.findChessboardCorners(image, (7,7))
    if not found:
        return None

    chessboard = read_chessboard(image, corners)
    return chessboard

