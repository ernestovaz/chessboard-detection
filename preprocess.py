import numpy as np
import cv2
import sys
import os
from pathlib import Path

GT_DIR = 'ground_truth'

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


def get_output_filename(directory):
    count = 0
    while os.path.exists(f'{directory}/gt{count}.png'):
        count += 1
    return f'{directory}/gt{count}.png'


# Creates ground truth image files
# We can later create directories for different themes too
#
def save_images(chessboard, theme_name):
    for piece_name, position_list in POSITIONS.items():
        directory = f'{GT_DIR}/{theme_name}/{piece_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for position in position_list:
            image = chessboard[position]
            edge_map = cv2.Canny(image, 0, 255)
            filename = get_output_filename(directory)
            cv2.imwrite(filename, edge_map)


def zero_pad_image(img, shape):
    if img.shape == shape:
        return img

    padded_img = np.full(shape, (0,0,0), dtype=np.uint8)

    x_center = (shape[1] - img.shape[1]) // 2
    y_center = (shape[0] - img.shape[0]) // 2
    padded_img[y_center:y_center+img.shape[0],
             x_center:x_center+img.shape[1]] = img

    return padded_img

def get_avg_shape(ground_truths_path: str):
    dim1 = 0
    dim2 = 0
    gt_images = os.listdir(ground_truths_path)

    for gt_image_name in gt_images:
        img = cv2.imread(ground_truths_path + gt_image_name)
        dim1 = max(dim1, img.shape[0])
        dim2 = max(dim2, img.shape[1])

    output_shape = (max(dim1, dim2), max(dim1, dim2), 3)
    imgs_sum = np.zeros(output_shape)

    for gt_image_name in gt_images:
        gt_image = cv2.imread(ground_truths_path + gt_image_name)
        gt_image = zero_pad_image(gt_image, output_shape)
        imgs_sum += gt_image

    return imgs_sum/len(gt_images)

def calculate_averages(theme_name):
    for piece in POSITIONS:
        piece_dir = f'{GT_DIR}/{theme_name}/{piece}/'
        if os.path.isdir(piece_dir):
            avg = get_avg_shape(piece_dir)
            cv2.imwrite(f'{piece_dir}avg.png', avg)

def main():
    filename = sys.argv[1]
    img = cv2.imread(filename)
    theme_name = Path(filename).stem
    found, corners = cv2.findChessboardCorners(img, (7,7))
    if found:
        chessboard = read_chessboard(img, corners)
        save_images(chessboard, theme_name)
        calculate_averages(theme_name)


if __name__ == '__main__':
    main()



