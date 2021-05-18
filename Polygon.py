import cv2
import numpy as np
from techsomed1 import Line2D
import math


class Polygon:
    def __init__(self):
        self.m_pol = []
        self.m_lines = []
        self.m_xo, self.m_yo = None, None
        self.m_img = None
    def set(self, pol):
        assert len(pol) > 0
        #assert isinstance(pol[0], cv2.)
        N = len(pol)
        p0 = pol[0]
        pn = pol[N-1]
        if p0[0] == pn[0] and p0[1] == pn[1]:
            self.m_pol = np.zeros((N-1,2), dtype=pol.dtype)
            self.m_pol[:,:] = pol[0:N-1,:]
        else:
            self.m_pol = pol
        self.build_lines()
    def move(self, delx, dely):
        pol = self.m_pol
        N = len(pol)
        for k in range(N):
            pol[k][0] += delx
            pol[k][1] += dely
        self.m_pol = pol
        self.build_lines()

    def set_from_bin_image(self, img):
        #ret, thresh = cv.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cur_area = -1
        LC = len(contours)
        lpol = []
        for k in range(LC):
            pol = Polygon()
            ck = contours[k]
            ck = np.squeeze(ck)
            pol.set(ck)
            ar = pol.get_area()
            if ar > cur_area:
                cur_area = ar
                pol = pol
        if cur_area>0:
            self.set(pol.m_pol)

        self.m_img = img
        idxs = img > 0
        ar = np.sum(idxs)
        return cur_area>0

    def build_lines(self):
        self.m_lines = []
        xo,yo = 0,0
        N = len(self.m_pol)
        for k in range(N):
            ln = Line2D.Line2D()
            kn = (k+1) % N
            ln.set(self.m_pol[k][0], self.m_pol[k][1],self.m_pol[kn][0], self.m_pol[kn][1])
            xo += self.m_pol[k][0]
            yo += self.m_pol[k][1]
            self.m_lines.append(ln)
        self.m_xo = xo/N
        self.m_yo = yo/N

    def get_area(self):
        LEN = len(self.m_pol)
        area = 0
        pol = self.m_pol
        for k in range(LEN):
            kn = (k + 1) % LEN
            area += (pol[kn][0] + pol[k][0]) * (pol[kn][1] - pol[k][1])

        if area < 0:
            area *= -1
        return area / 2.

    def get_intersection(self, x, y):
        ln = Line2D.Line2D()
        ln.set(self.m_xo,self.m_yo, self.m_xo + x, self.m_yo + y)
        max_dist = -1
        xo,yo = None, None
        for k in range(len(self.m_lines)):
            flg, xo,yo = ln.get_intersection(self.m_lines[k])
            if flg:
                x,y = xo,yo
                break
        return xo,yo


    def get_desc_points(self, delAng):
        # assumes angle in degrees
        delAng = delAng*math.pi/180.
        max_ang = math.pi * 2
        pnts = []
        L = 10e6
        ang = 0
        while ang < math.pi*2:
            dx,dy = math.cos(ang)*L, math.sin(ang)*L
            ang += delAng
            xo,yo = self.get_intersection(dx,dy)
            if xo is None or yo is None:
                xo,yo = 0, 0
            pnts.append((xo,yo))
        return pnts

    def display_vectors(self,pnts, img):
        start_point = (0, 0)

        # Green color in BGR
        color = (0, 255, 0)

        # Line thickness of 9 px
        thickness = 2
        img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for p in pnts:
            end_point = (int(round(p[0])), int(round(p[1])))
            cv2.line(img, start_point, end_point,color, thickness)
        # Displaying the image
        cv2.imshow('debug-features', img)
        cv2.waitKey(1)
    def get_dice(self, pol):
        dice = None
        if self.m_img is not None and pol.m_img is not None:
            iimg = np.logical_and(self.m_img, pol.m_img)
            scommon = np.sum(iimg!=0)
            s1 = np.sum(self.m_img!=0)
            s2 = np.sum(pol.m_img!=0)
            dice = 2*scommon/(s1+s2)
        return dice
    def sigmoid(self, val, mean=0.8, scale=10):
        val = (val - mean)*scale
        val = 1./(1+math.exp(-val))
        return val
    def get_pol_corr(self, pol, delAng=5, dbgImg=None):
        spnts = self.get_desc_points(delAng)
        if dbgImg is not None:
            self.display_vectors(spnts, dbgImg)
        opnts = pol.get_desc_points(delAng)
        m1 = np.array(spnts)
        m2 = np.array(opnts)

        m1 = m1.reshape((1,m1.size))
        m2 = m2.reshape((1,m2.size))
        m1m2 = np.sum(m1 * m2)
        m1m1 = math.sqrt(np.sum(m1*m1))
        m2m2 = math.sqrt(np.sum(m2*m2))
        rao = m1m2 / (m1m1 * m2m2)
        L = m1.shape[1]
        m1m2 = np.mean(np.abs(m1-m2))*2
        #m1m2 /= L
        #l2dist = np.mean(np.sqrt((m1-m2) * (m1-m2)))
        l2dist = 0
        max_dist = 0
        for i in range(0,L-2,2):
            dx,dy = m1[0,i]-m2[0,i], m1[0,i+1] - m2[0,i+1]
            cur_dist = math.sqrt(dx*dx + dy*dy)
            l2dist += cur_dist
            if cur_dist>max_dist:
                max_dist = cur_dist

        l2dist = l2dist*2/L
        #l2dist /= L
        #l2dist = math.sqrt(l2dist)
        dice = self.get_dice(pol)
        #rao2 = pow(rao,4)
        rao2 = self.sigmoid(rao,mean=0.9)
        return rao, rao2, m1m2, l2dist, dice, max_dist


def move_img(img, delx,dely):
    H,W = img.shape[:]
    nimg = np.zeros((H,W), np.uint8)
    for y in range(H):
        ny = y + dely
        if ny<0 or ny>=H:
            continue
        for x in range(W):
            nx = x + delx
            if nx<0 or nx>=W:
                continue
            nimg[ny,nx] = img[y,x]
    return  nimg

def test_move(file1, file2):
    img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
    pol1 = Polygon()
    pol2 = Polygon()
    pol1.set_from_bin_image(img1)
    pol2.set_from_bin_image(img2)
    rao, rao2, mdist, l2dist, dice = pol1.get_pol_corr(pol2, dbgImg=img1)
    print(f'rao={rao}, rao2={rao2} dice={dice}, manhatan_dist={mdist}, l2dist={l2dist}')
    delta = 20
    max_move = 50
    cv2.imshow('image1', img1)
    for mv in range(-max_move, max_move, 10):
        image2 = img2.copy()
        image2 = move_img(image2, mv, mv)
        pol2.set_from_bin_image(image2)
        # pol2.move(mv,mv)
        rao, rao2, mdist, l2dist, dice = pol1.get_pol_corr(pol2)
        print(f'rao after move({mv},{mv}) rao={rao}, rao2={rao2} dice={dice} manhatan_dist={mdist}, l2dist={l2dist}')
        cv2.imshow('images', image2)
        cv2.waitKey(100)

def test_2imgs(dir,file1, file2):
    res_file = dir + '/' + file1 + '_' + file2
    file1 = dir + '/' + file1
    file2 = dir + '/' + file2
    img1 = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print(f'Failed to read image: { file1}\n{file2}')
        return [-1, -1, -1, -1, -1, -1]
    pol1 = Polygon()
    pol2 = Polygon()
    pol1.set_from_bin_image(img1)
    pol2.set_from_bin_image(img2)
    rao, rao2, mdist, l2dist, dice, max_dist = pol1.get_pol_corr(pol2, dbgImg=img1)
    img_stk = np.dstack((img1,img1,img2))

    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (5, 30)

    # fontScale
    fontScale = 0.6

    # Blue color in BGR
    color = (255, 255, 255)

    # Line thickness of 2 px
    thickness = 1

    # Using cv2.putText() method
    txt_lst = ['rao:', rao, ' rao2:', rao2, ' mdist:', mdist, ' l2dist:', l2dist, ' dice:', dice , ' max_dist:', max_dist]
    txt = ''
    txt = ''
    for s in txt_lst:
        if isinstance(s,str):
            txt += str(s)
        else:
            st = '{:.3f}'.format(s)
            txt += st

    dim = (800,800)
    img_stk = cv2.resize(img_stk, dim, interpolation=cv2.INTER_AREA)
    image = cv2.putText(img_stk, txt, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(res_file, image)
    cv2.imshow('statistics', image)
    cv2.waitKey(1000)
    print(f'result:{txt}')
    #s = input('Enter any key to end:')
    return [rao, rao2, mdist, l2dist, dice, max_dist]

def test_from_cfg(dir,filename):
    count = 0
    stat = []
    files = []
    filename = dir + '/' + filename
    with open(filename) as file:
        while True:
            f1 = file.readline()
            f2 = file.readline()
            if not f2:
                break
            f1 = f1.strip('\n')
            f2 = f2.strip('\n')
            files.append((f1, f2))
            count += 1
    if count:
        for f in files:
            ll = test_2imgs(dir, f[0], f[1])
            stat.append(ll)
        arr = np.array(stat, dtype=np.float)
        mn = np.mean(arr, axis=1)
        ss = np.std(arr, axis=1)
        L = len(stat)
        ssm = ss/math.sqrt(L-1)
        resname = filename + '_res.txt'
        with open(resname, "w") as file:
            L = mn.shape[0]
            file.write('\trao,\t rao2,\t mdist,\t l2dist,\t dice,\t max_dist')
            # st = '{:.3f}'.format(s)
            lstr = ['{:.3f} '.format(mn[i]) for i in range(L) ]
            file.write(f'\n mean: {lstr}')

            lstr = ['{:.3f} '.format(ss[i]) for i in range(L)]
            file.write(f'\n std : {lstr}')

            lstr = ['{:.3f} '.format(ssm[i]) for i in range(L)]
            file.write(f'\n mstd: {lstr}')
            lstr = ''
            cf = 2
            for k in range(L):
                lstr += '[{:.3f}'.format(mn[k]-cf*ssm[k])
                lstr += ', {:.3f}]\t'.format(mn[k]+cf*ssm[k])
            file.write('\nBetween groups confidence intervals:\n')
            file.write(lstr)

# Unit test
def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--file1', dest = 'file1', type = str, required=True)
    ap.add_argument('--file2', dest='file2', type=str, required=True)
    ap.add_argument('--dir', dest='dir', type=str, required=True)
    ap.add_argument('--cfg', dest='cfg', type=str, default='', help='Config directory in the directory dir')
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.cfg != '':
        test_from_cfg(args.dir, args.cfg)
    else:
        test_2imgs(args.dir, args.file1, args.file2)
