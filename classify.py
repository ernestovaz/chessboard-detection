import os
import numpy as np
import cv2
GT_PAWN_DIR = 'ground_truth/pawn/'
GT_ROOK_DIR = 'ground_truth/rook/'

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

def main():
    ans = get_avg_shape(GT_PAWN_DIR)
    cv2.imwrite(GT_PAWN_DIR + 'avg.png', ans)

if __name__ == '__main__':
    main()