import os
import numpy as np
import cv2
import math

GT_DIR = 'ground_truth'
MAX = 10000000
INPUT_IMAGE = 'ground_truth/bishop/gt2.png'

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

def sum_squared_diff(img1, img2):
    sum = 0
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            sum+= math.sqrt(pow(img1[y][x][0] - img2[y][x][0],2))
            sum+= math.sqrt(pow(img1[y][x][1] - img2[y][x][1],2))
            sum+= math.sqrt(pow(img1[y][x][2] - img2[y][x][2],2))

    return sum/(img1.shape[0]*img1.shape[1])

def calc_dist(img1, img2):
    img1 = zero_pad_image(img1, img2.shape)
    return sum_squared_diff(img1, img2)

def classify(img):
    min_dist = MAX
    label = "pawn"
    directories = os.listdir(GT_DIR)
    for piece in directories:
        avg_img_pth = f'{GT_DIR}/{piece}/avg.png'
        if os.path.exists(avg_img_pth) and piece != '.DS_Store':
            avg_img = cv2.imread(avg_img_pth)
            print(avg_img_pth, avg_img.shape)
            dist = calc_dist(img, avg_img)
            if dist < min_dist:
                min_dist = dist
                label = piece
    return label

def main():
    directories = os.listdir(GT_DIR)
    for piece in directories:
        if piece != '.DS_Store':
            piece_dir = f'{GT_DIR}/{piece}/'
            ans = get_avg_shape(piece_dir)
            cv2.imwrite(f'{piece_dir}avg.png', ans)

    input_image = cv2.imread(INPUT_IMAGE)
    label = classify(input_image)
    print('')
    print('Image label:', label)
    print('')

if __name__ == '__main__':
    main()
