import numpy as np
import tensorflow as tf
eps = 1e-9
def Strategy(cu):
    # c = np.minimum(f1,f2)
    c1 = cu[:, :, :, 0:32]
    c2 = cu[:, :, :, 32:64]
    u1 = cu[:, :, :, 64:96]
    u2 = cu[:, :, :, 96:128]

    w=abs(c1)/(abs(c1)+abs(c2)+eps)
    c = w * c1 + (1 - w) * c2
    u = np.where(abs(u1)>abs(u2), u1, u2)
    # u = np.where(inde > 0, (u1 - np.min(u1)) / (np.max(u1) - np.min(u1)), (u2 - np.min(u2)) / (np.max(u2) - np.min(u2)))
    # w1 = ((u1 - np.min(u1)) / (np.max(u1) - np.min(u1))) / (((u1 - np.min(u1)) / (np.max(u1) - np.min(u1))) + (
    #         (u2 - np.min(u2)) / (np.max(u2) - np.min(u2))) + eps)
    # u = w1 * ((u1 - np.min(u1)) / (np.max(u1) - np.min(u1))) + (1 - w1) * ((u2 - np.min(u2)) / (np.max(u2) - np.min(u2)))
    # u = np.maximum((u1 - np.min(u1)) / (np.max(u1) - np.min(u1)),(u2 - np.min(u2)) / (np.max(u2) - np.min(u2)))
    return c,u
