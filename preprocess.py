import numpy as np
import cv2
import sys
import os
from pathlib import Path

import chessboard

GT_DIR = 'ground_truth'
IMG_SIZE = 32

def get_output_filename(directory, theme_name):
    count = 0
    while os.path.exists(f'{directory}/{theme_name}{count}.png'):
        count += 1
    return f'{directory}/{theme_name}{count}.png'

def get_edges(image):
    return cv2.Canny(image, 0, 255)


# Creates ground truth image files
# We can later create directories for different themes too
#
def save_images(board, theme_name):
    for piece_name, position_list in chessboard.POSITIONS.items():
        directory = f'{GT_DIR}/{piece_name}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        for position in position_list:
            image = board[position]
            downscaled = downscale_image(image)
            filename = get_output_filename(directory, theme_name)
            cv2.imwrite(filename, downscaled)


def zero_pad_image(img, shape):
    if img.shape == shape:
        return img

    padded_img = np.full(shape, 0, dtype=np.uint8)

    x_center = (shape[1] - img.shape[1]) // 2
    y_center = (shape[0] - img.shape[0]) // 2
    padded_img[y_center:y_center+img.shape[0],
             x_center:x_center+img.shape[1]] = img

    return padded_img


def downscale_image(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), cv2.INTER_LINEAR)


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
        gt_image = downscale_image(gt_image)
        imgs_sum += gt_image

    return imgs_sum/len(gt_images)

def calculate_averages(theme_name):
    for piece in chessboard.POSITIONS:
        piece_dir = f'{GT_DIR}/{theme_name}/{piece}/'
        if os.path.isdir(piece_dir):
            avg = get_avg_shape(piece_dir)
            cv2.imwrite(f'{piece_dir}avg.png', avg)

def main():
    filename = sys.argv[1]
    print('File: ', filename, end=' ')
    img = cv2.imread(filename)
    theme_name = Path(filename).stem

    board = chessboard.from_image(img)
    if board is not None:
        save_images(board, theme_name)
        print('OK!')


if __name__ == '__main__':
    main()



