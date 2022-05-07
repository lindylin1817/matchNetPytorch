import cv2
import numpy as np
import os

if __name__ == "__main__":
    pair_output_a = "./after_pair/a/"
    pair_output_b = "./after_pair/b/"
    img_a_path = "./to_tile/a6.jpg"
    img_b_path = "./to_tile/b6.jpg"
    del_list = os.listdir(pair_output_a)
    for f in del_list:
        file_path = os.path.join(pair_output_a, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    del_list = os.listdir(pair_output_b)
    for f in del_list:
        file_path = os.path.join(pair_output_b, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    tmp_str = img_a_path[img_a_path.rindex("/") + 1:]
    filename_prefix = tmp_str[:tmp_str.rindex(".")]
    print(filename_prefix)
    tile_pixels = 200
    overlap_ratio = 0.3
    tile_pixels_overlap = int(200 * (1 - overlap_ratio))
    region_points_ratio = [[0.4, 0], [1, 1]]
#    region_points_ratio = None
    img_a = cv2.imread(img_a_path)
    img_b = cv2.imread(img_b_path)

    (image_height, image_width) = img_a.shape[:2]

    if region_points_ratio is None:
        region_points_ratio = [[0, 0], [1, 1]]

    region_points = np.zeros((len(region_points_ratio), 2), dtype=int)
    for i in range(len(region_points_ratio)):
        region_points[i][0] = int(region_points_ratio[i][0] * image_width)
        region_points[i][1] = int(region_points_ratio[i][1] * image_height)
    print(region_points)

    i = 0
    for x in range(region_points[0][0], region_points[1][0], tile_pixels_overlap):
        for y in range(region_points[0][1], region_points[1][1], tile_pixels_overlap):
            if ((region_points[1][0] - x) < tile_pixels) or ((region_points[1][1] - y)<tile_pixels):
                break
            cropped_a = img_a[y:y+tile_pixels, x:x+tile_pixels]
            cropped_b = img_b[y:y + tile_pixels, x:x + tile_pixels]
            i += 1
            filename = filename_prefix + "_" + str(i) + ".jpg"
            print(pair_output_a+filename)
            cv2.imwrite(pair_output_a+filename, cropped_a)
            cv2.imwrite(pair_output_b+filename, cropped_b)



