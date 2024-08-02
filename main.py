import cv2 as cv
import numpy as np


# minimum number of matches that have to be found in one frame
MIN_MATCH_COUNT = 10


def main():
    # create SIFT operator
    sift = cv.SIFT_create()
    # create FlannBased matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    # load reference image
    ref = cv.imread("steve.jpg", 0)
    # size of reference image
    h_ref, w_ref = ref.shape[:2]
    # detect key-points and compute descriptors in reference image
    kp_ref, des_ref = sift.detectAndCompute(ref, None)
    # load source video
    vc = cv.VideoCapture("steve.mp4")
    f_video = int(vc.get(7))
    video = np.empty((f_video, h_ref, w_ref, 3), dtype=np.uint8)
    for f in range(f_video):
        ret, frame = vc.read()
        # resize source video
        video[f] = cv.resize(frame, (w_ref, h_ref))
    vc.release()

    # start capturing video
    vc = cv.VideoCapture(0)
    f = 0
    while True:
        # read frame
        ret, frame = vc.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # size of frame (convert to int)
        w_frame, h_frame = int(vc.get(3)), int(vc.get(4))

        # detect key-points and compute descriptors in current frame
        kp_frame, des_frame = sift.detectAndCompute(
            cv.cvtColor(frame, cv.COLOR_BGR2GRAY), None)
        if des_frame is None:
            print("Not clear enough for getting descriptor.")
            continue

        # match frame descriptors with reference image descriptors
        matches = flann.knnMatch(des_ref, des_frame, k=2)
        # store good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # check whether number of good matches greater than threshold
        if len(good_matches) < MIN_MATCH_COUNT:
            print("Not enough matches are found - {}/{}".format(
                len(good_matches), MIN_MATCH_COUNT))
        else:
            # find perspective transformation
            src_pts = np.float32([kp_ref[m.queryIdx].pt
                                  for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt
                                  for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

            # warp source video
            try:
                warped_video = cv.warpPerspective(
                    video[f], M, (w_frame, h_frame))
                f = (f + 1) % f_video

                # construct a mask covering the warped source image
                mask = np.zeros((h_frame, w_frame), dtype=np.uint8)
                corners = np.float32([[-5, -5],
                                      [-5, h_ref + 4],
                                      [w_ref + 4, h_ref + 4],
                                      [w_ref + 4, -5]]
                                     ).reshape(-1, 1, 2)
                warped_corners = cv.perspectiveTransform(corners, M)
                cv.fillConvexPoly(mask, np.int32(warped_corners),
                                  (1, 1, 1), cv.LINE_AA)
                mask = np.dstack([mask] * 3)  # mask need to be 3-channel

                # image overlay
                frame = cv.add(cv.multiply(mask, warped_video),
                               cv.multiply(frame, 1 - mask))
            except:
                pass

        cv.imshow("Camera Capture", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
