import tensorflow as tf

from m_encoder import m_Encoder
from m_decoder import m_Decoder
from s_encoder import s_Encoder
from s_decoder import s_Decoder

EPSILON = 1e-9

class DenseFuseNet(object):

    def __init__(self, sco):
        self.m_encoder = m_Encoder(sco)
        self.m_decoder = m_Decoder(sco)
        self.s_encoder = s_Encoder(sco)
        self.s_decoder = s_Decoder(sco)

    def test(self,target_features):
        generated_img = self.m_decoder.decode(target_features)
        return generated_img

    def transform_test_part1(self, img1,img2):
        # encode image
        img1 = (img1 - tf.reduce_min(img1)) / (tf.reduce_max(img1) - tf.reduce_min(img1))
        img2 = (img2 - tf.reduce_min(img2)) / (tf.reduce_max(img2) - tf.reduce_min(img2))
        f11, f12, f13, f14 = self.m_encoder.encode(img1)
        f21, f22, f23, f24 = self.m_encoder.encode(img2)
        return f11,f12,f13,f14,f21,f22,f23,f24

    def transform_test_part2(self,f11,f12,f13,f14,f21,f22,f23,f24):
        cu1 = self.s_encoder.encode(f11, f21)
        cu2 = self.s_encoder.encode(f12, f22)
        cu3 = self.s_encoder.encode(f13, f23)
        cu4 = self.s_encoder.encode(f14, f24)
        return cu1,cu2,cu3,cu4

    def transform_test_part3(self, fc1,fu1,fc2,fu2,fc3,fu3,fc4,fu4):
        f1 = self.s_decoder.decode(fc1, fu1)
        f2 = self.s_decoder.decode(fc2, fu2)
        f3 = self.s_decoder.decode(fc3, fu3)
        f4 = self.s_decoder.decode(fc4, fu4)
        return f1,f2,f3,f4

    def transform_test_part4(self,f1,f2,f3,f4):
        target_features = tf.concat([f1, f2, f3, f4], axis=3)
        self.target_features = target_features
        generated_img = self.m_decoder.decode(target_features)
        return generated_img

    def transform_recons_part1(self, img):
        img=(img-tf.reduce_min(img))/(tf.reduce_max(img)-tf.reduce_min(img))
        f11, f12, f13, f14 = self.m_encoder.encode(img)
        target_features1 = tf.concat([f11,f12,f13,f14],axis=3)
        self.target_features1 = target_features1
        _f = self.m_decoder.decode(target_features1)
        return f11, f12, f13, f14, _f

    def transform_recons_part2(self, f11,f12,f13,f14,f21,f22,f23,f24):
        f11 = f11 / (tf.reduce_max(f11) + EPSILON)
        f12 = f12 / (tf.reduce_max(f12) + EPSILON)
        f13 = f13 / (tf.reduce_max(f13) + EPSILON)
        f14 = f14 / (tf.reduce_max(f14) + EPSILON)
        f21 = f21 / (tf.reduce_max(f21) + EPSILON)
        f22 = f22 / (tf.reduce_max(f22) + EPSILON)
        f23 = f23 / (tf.reduce_max(f23) + EPSILON)
        f24 = f24 / (tf.reduce_max(f24) + EPSILON)
        cu1 = self.s_encoder.encode(f11, f21)
        cu2 = self.s_encoder.encode(f12, f22)
        cu3 = self.s_encoder.encode(f13, f23)
        cu4 = self.s_encoder.encode(f14, f24)

        _f11 = self.s_decoder.decode(cu1[:,:,:,0:32], cu1[:, :, :, 64:96])
        _f21 = self.s_decoder.decode(cu1[:,:,:,32:64], cu1[:, :, :, 96:128])
        _f12 = self.s_decoder.decode(cu2[:,:,:,0:32], cu2[:, :, :, 64:96])
        _f22 = self.s_decoder.decode(cu2[:,:,:,32:64], cu2[:, :, :, 96:128])
        _f13 = self.s_decoder.decode(cu3[:,:,:,0:32], cu3[:, :, :, 64:96])
        _f23 = self.s_decoder.decode(cu3[:,:,:,32:64], cu3[:, :, :, 96:128])
        _f14 = self.s_decoder.decode(cu4[:,:,:,0:32], cu4[:, :, :, 64:96])
        _f24 = self.s_decoder.decode(cu4[:,:,:,32:64], cu4[:, :, :, 96:128])

        _f11 = _f11 * tf.reduce_max(f11)
        _f12 = _f12 * tf.reduce_max(f12)
        _f13 = _f13 * tf.reduce_max(f13)
        _f14 = _f14 * tf.reduce_max(f14)
        _f21 = _f21 * tf.reduce_max(f21)
        _f22 = _f22 * tf.reduce_max(f22)
        _f23 = _f23 * tf.reduce_max(f23)
        _f24 = _f24 * tf.reduce_max(f24)
        return cu1,cu2,cu3,cu4,_f11,_f12,_f13,_f14,_f21,_f22,_f23,_f24



