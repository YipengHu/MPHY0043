
#This tutorial is adapted from [Point cloud classification with PointNet](https://keras.io/examples/vision/pointnet/) with TensorFlow and Keras.


import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import utils
from networks import pointnet


tf.random.set_seed(43)


## dataset - the "motion class"
DATA_DIR = 'datasets/'
file_paths = {"FILE_TRAIN" : "nodes_train.npy", "FILE_TEST" : "nodes_test.npy", "FILE_TRIS" : "tris.npy"}
for f,fn in file_paths.items():
    file_paths[f] = DATA_DIR+fn

nodes_motion_train = np.load(file_paths['FILE_TRAIN']).astype(np.float32)
nodes_motion_test = np.load(file_paths['FILE_TEST']).astype(np.float32)
tris = np.load(file_paths['FILE_TRIS']).astype(np.float32)

# normalise the data
ref_id = 0
nodes_ref = nodes_motion_train[...,ref_id]
disp = - nodes_ref.min(0) # use the first one as reference
scale = 1 / (nodes_ref.max(0)-nodes_ref.min(0)).max()
nodes_ref = utils.normalise_pointset(nodes_ref[...,None], scale, disp)
nodes_motion_train = utils.normalise_pointset(np.delete(nodes_motion_train,ref_id,axis=2), scale, disp)
nodes_motion_test = utils.normalise_pointset(nodes_motion_test, scale, disp)


## simulate the "affine class"
affine_scale = 0.2
nodes_affine_train = utils.random_affine_transform_pointset(nodes_ref, num=nodes_motion_train.shape[0], scale=affine_scale)
nodes_affine_test = utils.random_affine_transform_pointset(nodes_ref, num=nodes_motion_test.shape[0], scale=affine_scale)


## add noise for the "noise class"
noise_std = 0.005
nodes_noise_train = nodes_ref + tf.random.normal(nodes_motion_train.shape,stddev=noise_std)
nodes_noise_test = nodes_ref + tf.random.normal(nodes_motion_test.shape,stddev=noise_std)


## Q: how small affine_scale and noise_std can be detected?


## plot all
idx = np.random.choice(nodes_motion_train.shape[0])
fig = plt.figure(figsize=[12.8,9.6], tight_layout=True)
ax = fig.add_subplot(2,2,1, projection='3d')
ax.plot_trisurf(nodes_ref[0,:,0], nodes_ref[0,:,1], nodes_ref[0,:,2], triangles=tris)
ax.title.set_text('Reference')
ax = fig.add_subplot(2,2,2, projection='3d')
ax.plot_trisurf(nodes_motion_train[idx,:,0], nodes_motion_train[idx,:,1], nodes_motion_train[idx,:,2], triangles=tris)
ax.title.set_text('Motion')
ax = fig.add_subplot(2,2,3, projection='3d')
ax.plot_trisurf(nodes_affine_train[idx,:,0], nodes_affine_train[idx,:,1], nodes_affine_train[idx,:,2], triangles=tris)
ax.title.set_text('Affine')
ax = fig.add_subplot(2,2,4, projection='3d')
ax.plot_trisurf(nodes_noise_train[idx,:,0], nodes_noise_train[idx,:,1], nodes_noise_train[idx,:,2], triangles=tris)
ax.title.set_text('Noise')
# plt.show()
plt.savefig('example_motion.jpg',bbox_inches='tight')
print('Examples of different types of motion saved.')


## putting all points together
train_points = tf.concat([nodes_motion_train,nodes_affine_train,nodes_noise_train],axis=0)
train_labels = tf.cast(tf.concat([tf.zeros(nodes_motion_train.shape[0]),tf.ones(nodes_affine_train.shape[0]),tf.ones(nodes_noise_train.shape[0])*2],axis=0),tf.int8)

test_points = tf.concat([nodes_motion_test,nodes_affine_test,nodes_noise_test],axis=0)
test_labels = tf.cast(tf.concat([tf.zeros(nodes_motion_test.shape[0]),tf.ones(nodes_affine_test.shape[0]),tf.ones(nodes_noise_test.shape[0])*2],axis=0),tf.int8)

CLASS_MAP = {0:'motion', 1:'affine', 2:'noise'}
BATCH_SIZE = 32


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(utils.augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)


## set up the pointnet model
model = pointnet(num_points=train_points.shape[1], num_class=len(CLASS_MAP))
model.summary()


### Train 
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)


## Visualize predictions
data = test_dataset.take(1)

points, labels = list(data)[0]
points = points[:8, ...]
labels = labels[:8, ...]

# run test data through model
preds = model.predict(points)
preds = tf.math.argmax(preds, -1)

points = points.numpy()

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(8):
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
    ax.set_title(
        "pred: {:}, label: {:}".format(
            CLASS_MAP[preds[i].numpy()], CLASS_MAP[labels.numpy()[i]]
        )
    )
    ax.set_axis_off()
#plt.show()
plt.savefig('example_results.jpg',bbox_inches='tight')
print('Example classification results saved.')
