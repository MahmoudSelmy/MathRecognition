import cv2
import numpy as np

def cv_size(img):
    return tuple(img.shape[1::-1])  # returns (width,height) as opencv expects this format as size parameters

def get_contours(img):

    # blurring the image
    blurred_image = cv2.GaussianBlur(img,(3,3),0)
    # applying adaptive threshold to the blurred image
    adaptive_thresholded = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    inverted_image = cv2.bitwise_not(adaptive_thresholded)

    # invert to let text be in white
    img2 = inverted_image.copy()   # rows * columns
    #print(len(img2))

    points_list = np.argwhere(img2.T==255)  # to get x,y not y,x tuples--> x:right , y:down

    rec = cv2.minAreaRect(points_list)  # ((x,y), (h,w), theta) where theta is the negative the angle with x axis
    theta = rec[-1]
    if theta < -45 :
        theta += 90
        print("theta += 90")


    box = cv2.boxPoints(rec)   # list of length 4 , each being a list of x,y vertex

    rot_matrix = cv2.getRotationMatrix2D(rec[0],theta,1)
    rotated = cv2.warpAffine(img2, rot_matrix, cv_size(img2), cv2.INTER_CUBIC)

    h,w = rec[1]
    if rec[-1] < -45:
        print("rec[-1] < -45")
        rec = (rec[0], (w,h), rec[-1]) # 3ashan tabe3y ya3ny


    cropped = cv2.getRectSubPix(rotated, (int(h),int(w)), tuple(map(int,rec[0])))

    cropped2 = cropped.copy()
    cropped2_rgb = cv2.cvtColor(cropped2, cv2.COLOR_GRAY2RGB)
    cropped3 = cropped.copy()
    cropped3_rgb = cv2.cvtColor(cropped3, cv2.COLOR_GRAY2RGB)

    _, contours, hierarchy = cv2.findContours(cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS, offset=(0,0))


    ### Approximate contours to polygons to get bounding rectangles and circles
    contours_poly = list(map(lambda x : cv2.approxPolyDP(x, 3, True), contours))
    print(len(contours_poly))

    valid_contours = []
    bound_rects = []
    for i, contour in enumerate(contours_poly):
        rec_contour = cv2.boundingRect(contour)
        x1,y1,w1,h1 = rec_contour
        print(rec_contour)
        if w1*h1 < 100:  # ignore small contours
            continue
        inside = False

        for j, other_contour in enumerate(contours_poly):
            if i == j:
                continue

            rec_contour_2 = cv2.boundingRect(other_contour)
            x2, y2, w2, h2 = rec_contour_2
            if (h2*w2 < 100 or h2*w2 < h1*w1):
                continue
            if (x1 > x2 and x1+w1 < x2+w2 and y1 > y2 and y1+h1 < y2+h2):
                inside = True
        if inside:
            continue
        valid_contours.append(contour)
        bound_rects.append(rec_contour)
        #top_left = (x1,y1)
        #bottom_right = (x1 + w1 , y1 + h1)
        #print(top_left)
        #print(bottom_right)
        #print("==")
        box_i = cv2.boxPoints(((x1+w1//2,y1+h1//2),(w1,h1),0.0))
        cv2.drawContours(cropped2, [box_i.astype(int)], 0, (255, 255, 255), 1)
        #cv2.rectangle(cropped2,top_left,bottom_right,(255,0,255))


    print(len(valid_contours))



    centers = []
    radiuses = []


    print("/////////////////////////////////////////////////////")
    return(cropped2)

if __name__ == "__main__":
    img = cv2.imread('test3.jpg', 0)
    img2 = get_contours(img)
    cv2.imshow('original image', img)
    cv2.imshow('image', img2)
    cv2.waitKey(0)
