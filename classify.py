import os
import cv2
import math

from preprocess import zero_pad_image

GT_DIR = 'ground_truth/chess_com_default'
MAX = 10000000
INPUT_IMAGE = 'ground_truth/chess_com_default/pawn/gt2.png'

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
    input_image = cv2.imread(INPUT_IMAGE)
    label = classify(input_image)
    print('')
    print('Image label:', label)
    print('')

if __name__ == '__main__':
    main()
