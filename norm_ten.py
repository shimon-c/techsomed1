import numpy as np
def naive_norm(ten, chan_axis=2):
    EPS = 1e-8
    mn = np.zeros((3,))
    ss = np.zeros((3,))
    if chan_axis == 2:
        H,W,C = ten.shape
        for c in range(C):
            for y in range(H):
                for x in range(W):
                    v = ten[y,x,c]
                    mn[c] += v
                    ss[c] += v*v
        ss /= (W*H)
        mn /= (W*H)
        ss = ss - mn*mn
        ss = np.sqrt(ss)
        ss += EPS
        ten = (ten-mn)/ss
    return ten

def norm_ten(ten, chan_axis=2):
    EPS = 1e-8
    if chan_axis == 2:
        H,W,C = ten.shape
        ten = ten.reshape((W*H,C))
        mn = np.mean(ten, axis=0)
        ss = np.std(ten, axis=0) + EPS
        ten = (ten - mn)/ss
        ten = ten.reshape((H,W,C))
    elif chan_axis==0:
        C, H, W = ten.shape
        ten = ten.reshape((C,W*H))
        mn = np.mean(ten, axis=1)
        mn = mn.reshape((3,1))
        ss = np.std(ten, axis=1) + EPS
        ss = ss.reshape((3,1))
        ten = (ten - mn)/ss
        ten = ten.reshape((C,H,W))
    return ten

ten1 = np.random.rand(5,5,3)
ten2 = np.random.rand(3,5,5)

ten1copy = ten1.copy()
ten1 = norm_ten(ten1, chan_axis=2)
ten2 = norm_ten(ten2, chan_axis=0)

ten1n = naive_norm(ten1copy,chan_axis=2)
vals = np.abs(ten1 - ten1n)
ids = vals > 1e-8
print(f'ids={ids}')