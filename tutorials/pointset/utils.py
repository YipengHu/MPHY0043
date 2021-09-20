
import tensorflow as tf

def normalise_pointset(ps, scale, disp):
    return (tf.transpose(ps,[2,0,1]) + disp[None,None,]) * scale


def random_affine_transform(num=1,scale=0.1):
    corners = tf.reshape(tf.stack(tf.meshgrid([0.,1.],[0.,1.],[0.,1.]),axis=3),[-1,3])[None,...]
    corners_new = corners + tf.random.uniform((num,8,3),-1,1)*scale
    return tf.linalg.lstsq(tf.repeat(corners,num,0),corners_new)


def random_affine_transform_pointset(ps,**kwargs):
    transforms = random_affine_transform(**kwargs)
    return tf.matmul(ps,transforms)

