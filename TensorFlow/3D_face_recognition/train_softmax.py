from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

# from models.inception_resnet_v2 import InceptionResNetV2
from models.resnet50 import ResNet50
# from tensorflow.keras.applications.resnet50 import ResNet50
from dataset import Dataset


if __name__ == '__main__':
    # TODO: argument parser should be added.
    # tf.compat.v1.enable_eager_execution()
    # tf.debugging.set_log_device_placement(True)

    data_root = '~/vggface3d_sm'
    # model_path = '../logs/inception-resnet-v2.h5'
    model_path = '../logs/resnet50-3d.h5'
    batch_size = 16         # larger batch_size might cause segmentation fault
    num_epochs = 1000       # Number of epochs to run.
    steps_per_epoch = 2500  # You must specify the `steps_per_epoch` 'cause the training dataset was repeated
    input_image_size = (182, 182)
    input_shape = input_image_size + (4, )  # Model input shape

    train_dataset = Dataset(os.path.join(data_root, 'train.csv'),
                            is_repeat=True, is_shuffle=True,
                            input_image_size=input_image_size, target_image_size=None,
                            batch_size=batch_size)
    eval_dataset = Dataset(os.path.join(data_root, 'eval.csv'),
                           is_shuffle=True,
                           input_image_size=input_image_size, target_image_size=None,
                           batch_size=batch_size)

    assert train_dataset.get_num_of_classes() == eval_dataset.get_num_of_classes(), \
        "#classes in the training set should equals to #classes in the evaluation set"

    num_of_classes = train_dataset.get_num_of_classes()

    print("Training   dataset size: %d" % len(train_dataset))
    print("Evaluation dataset size: %d" % len(eval_dataset))
    print("Number of classes: %d" % num_of_classes)

    # data-parallelism distributed training
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential([
            # InceptionResNetV2(input_shape=input_shape, include_top=False, pooling='avg'),
            ResNet50(input_shape=input_shape, include_top=False, pooling='avg', weights="imagenet"),
            tf.keras.layers.Dense(num_of_classes, activation='softmax', name='predictions'),
        ])

        # decay every epoch with a base of 0.99
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=steps_per_epoch,
            decay_rate=0.996,
            staircase=True)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        if os.path.exists(model_path):
            model.load_weights(model_path)
            print("Checkpoint loaded.")
        else:
            print("No checkpoint loaded. It will train from scratch.")
        model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=25, monitor='val_loss'),  # to avoid over-fitting
        tf.keras.callbacks.TensorBoard(log_dir='../logs'),
        tf.keras.callbacks.CSVLogger('../logs/training.log'),
    ]

    model.fit(train_dataset.ds,
              epochs=num_epochs, steps_per_epoch=steps_per_epoch,
              validation_data=eval_dataset.ds,
              callbacks=callbacks)

    test_dataset = Dataset(os.path.join(data_root, 'test.csv'),
                           input_image_size=input_image_size, target_image_size=None,
                           batch_size=batch_size)
    results = model.evaluate(test_dataset.ds)
    print('test loss, test acc:', results)

