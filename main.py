# This is a sample Python script.
import numpy as np
import cv2
import math


def warp_flow(img, flow):
    flow = -flow    # Invert flow sow that warp_flow(img0, calc_opt_flow(img0, img1)) ~= img1
    h, w = flow.shape[:2]
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img.astype(flow.dtype), flow, None, cv2.INTER_LINEAR)
    return res.astype(img.dtype)





def calculate_hommography_vector_field(img, mat):
    Y,X = img.shape[:2]
    #X,Y = in_img.shape
    ###################################
    # For tests
    #X,Y = 2,4
    #mat = np.zeros((3,3))
    #mat[0,2] = 5
    #mat[1,2] = 0.11
    #mat[2,1] = 1
    ###################################
    #rng = np.arange(Y)[:, np.newaxis]
    #rng1 = np.arange(Y)
    timg = np.zeros((3,X,Y))
    rx = np.arange(Y)
    rx = np.repeat(rx,X)
    rx = np.reshape(rx, (X,Y))
    timg[0,:,:] = rx
    ry = np.arange(X)
    ry = np.tile(ry,Y)

    ry = np.reshape(ry, (X,Y))
    timg[1,:,:] = ry
    timg = timg.reshape((3,Y*X))
    timg[-1,:] = 1
    rimg = np.matmul(mat, timg)
    # swap axes
    rimg = rimg.reshape((3,Y,X))
    rimg = np.swapaxes(rimg,0,1)
    rimg = np.swapaxes(rimg,1,2)
    rimg = rimg[:,:,0:2]
    rimg = rimg.astype(np.float32)
    mapx = rimg[:,:,1]
    mapy = rimg[:,:,0]
    # last swap between X and Y
    #rimg[:,:,0] = dely
    #rimg[:,:,1] = delx
    rimg = rimg.copy()
    rimg[:,:,0] = mapx
    rimg[:, :, 1] = mapy
    return rimg, mapx, mapy



def test_np(in_img, ang=10):
    ang = math.pi/90
    #ang = 0
    cs = math.cos(ang)
    sn = math.sin(ang)
    mat = np.eye(3)
    mat[0,0], mat[1,1] = cs,cs
    mat[1,0] = -sn
    mat[0,1] = sn
    mat[0,2] = 0
    mat[1,2] = 0
    mat[2,2] = 1
    # dsize = (in_img.shape[1], in_img.shape[0])
    wrapped_img = cv2.warpPerspective(in_img, mat, in_img.shape[1::-1])
    #X,Y = in_img.shape[1::-1]
    rimg, map_x, map_y = calculate_hommography_vector_field(in_img, mat)
    #flow = rimg[:,:,0:2]
    flow = rimg.astype(np.float32)
    #flow_img = cv2.remap(in_img, flow, None, cv2.INTER_LINEAR)
    #flow_img = cv2.remap(in_img, map_x, map_y, cv2.INTER_LINEAR)
    #flow_img = cv2.remap(in_img, map_y, map_x, cv2.INTER_LINEAR)
    flow_img = cv2.remap(in_img, map_x, map_y, cv2.INTER_LINEAR)
    flow_img2 = cv2.remap(in_img, flow, None, cv2.INTER_LINEAR)
    cv2.imshow('in_image', in_img)
    cv2.imshow('wrapped_img', wrapped_img)
    cv2.imshow('flow_img', flow_img)
    cv2.imshow('flow_img2', flow_img2)
    cv2.waitKey(1)
    return rimg


# Press the green button in the gutter to run the script.
#this is branch2
if __name__ == '__main__':
    img_name = 'c:/Users/shimon.TECHSOMED/SW/BioTrace/TestsDB/raw_data_xy/mayo_p01_c1/full_frames/frame1050.png'
    in_img = cv2.imread(img_name, flags=cv2.IMREAD_GRAYSCALE)
    # make sure to get different
    sty,edy = 100,500
    stx,edx = 300,500
    in_img = in_img[sty:edy, stx:edx]
    img = test_np(in_img)
    import sys

    print(sys.path)
    print('Enter any key to end:')
    s = input('Any key:')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
