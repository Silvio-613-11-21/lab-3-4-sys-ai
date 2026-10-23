import cv2


def preprocess_fingerprint(image_path):
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None, None

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    _, img_bin = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return img_bin, img_color


def match_fingerprints(img1_path, img2_path, markers=True, unit_markers=True, coef=0.7):
    img1_bin, img1_color = preprocess_fingerprint(img1_path)
    img2_bin, img2_color = preprocess_fingerprint(img2_path)

    if img1_bin is None or img2_bin is None:
        return 0, None, []

    orb = cv2.ORB_create()

    kp1, descriptors1 = orb.detectAndCompute(img1_bin, None)
    kp2, descriptors2 = orb.detectAndCompute(img2_bin, None)

    if descriptors1 is None or descriptors2 is None:
        return 0, None, []

    # knnMatch(k=2) — нужно для ratio-теста Лоу (coef=0.7 по умолчанию)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < coef * n.distance:
            good_matches.append(m)

    # Флаги отрисовки
    if markers and unit_markers:
        flags = cv2.DrawMatchesFlags_DEFAULT
    else:
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

    points = []
    for m in good_matches:
        x, y = kp1[m.queryIdx].pt
        points.append({'x': x, 'y': y})

    match_img = cv2.drawMatches(
        img1_color, kp1, img2_color, kp2, good_matches, None,
        flags=flags,
    )

    return len(good_matches), match_img, points