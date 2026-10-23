import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def preprocess_fingerprint(image_path):
    # Читаем цветную версию (для отрисовки)
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None, None

    # Читаем серую версию (для SIFT)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Бинаризация
    _, img_bin = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return img_bin, img_color


def match_fingerprints(img1_path, img2_path, markers = True, unit_markers = True, coef = 0.7 ):
    img1_bin, img1_color = preprocess_fingerprint(img1_path)
    img2_bin, img2_color = preprocess_fingerprint(img2_path)

    if img1_bin is None or img2_bin is None:
        return 0, None

    # SIFT по бинарным (серым) изображениям
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(img1_bin, None)
    kp2, des2 = sift.detectAndCompute(img2_bin, None)

    if des1 is None or des2 is None:
        return 0, None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < coef * n.distance]

    points = []
    for m in good_matches:
        x, y = kp1[m.queryIdx].pt
        points.append({
            'x': x, 
            'y': y 
        })


    flags = cv2.DrawMatchesFlags_DEFAULT 
    # if(markers and unit_markers):
    #     flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    if(not markers and unit_markers):
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS


    match_img = cv2.drawMatches(
        img1_color, kp1, img2_color, kp2, good_matches, None,
        flags=flags 
        # flags=cv2.DrawMatchesFlags_DEFAULT
        # //flags=cv2.DrawMatchesFlags
        # flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
    )
    return len(good_matches), match_img, points
 



# img1_path = "c:\\Users\\sarba\\Downloads\\20774.jpg"
# img2_path = "c:\\Users\\sarba\\Downloads\\09878809.jpg"
# # Example usage
# len_ , match_img = match_fingerprints(img1_path, img2_path)

# if match_img is None:
#     print("=(")
# else:
#     cv2.imshow("tst", match_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()