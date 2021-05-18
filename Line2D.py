import numpy as np
import cv2
import math

class Line2D:
    def __init__(self):
        self.m_dx, self.m_dy = 0,0
        self.m_xo = self.m_yo = None
        self.m_wx = self.m_wy = self.m_wo = 0

    def get_point_for_t(self, t) :
        xo = t * self.m_dx + self.m_xo
        yo = t * self.m_dy + self.m_yo
        return xo,yo

    def set(self,x1,y1,x2,y2):
        self.m_dx = x2 - x1
        self.m_dy = y2 - y1
        fac = math.sqrt(self.m_dx * self.m_dx + self.m_dy * self.m_dy);
        EPS = 1e-8
        if fac < EPS:
            raise ArithmeticError('Line2D.set bad line too short')
        self.m_xo = x1
        self.m_yo = y1
        self.m_wy = self.m_dx / fac
        self.m_wx = -self.m_dy / fac;
        self.m_wo = 0
        v1 = self.get_signed_dist(x1, y1)
        v2 = self.get_signed_dist(x2, y2)
        self.m_wo = -(v1 + v2) / 2.


    def get_intersection(self, ln):
        mat = np.zeros((2,2), np.float)
        EPS = 1e-8
        mat[0,:] = self.m_dx, -ln.m_dx
        mat[1,:] = self.m_dy, -ln.m_dy

        det = np.linalg.det(mat)
        rc = False
        xo, yo = None, None
        if det>-EPS and det<EPS:
            return rc, xo, yo
        imat = np.linalg.inv(mat)

        x1 = ln.m_xo - self.m_xo
        y1 = ln.m_yo - self.m_yo

        t = imat[0,0] * x1 + imat[0,1] * y1
        s = imat[1,0] * x1 + imat[1,1] * y1


        if s >= 0 and s <= 1 and t >= 0 and t <= 1:
            rc = True
            xo,yo = self.get_point_for_t(t)

        return rc,xo,yo

    def get_signed_dist(self, x, y):
        return self.m_wx * x + self.m_wy * y + self.m_wo

    def get_abs_dist(self, x,y):
        dst = self.get_signed_dist(x,y)
        if dst<0: dst= -1
        return dst

