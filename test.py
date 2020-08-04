import tensorflow as tf
#########################
## Hardware management ##
#########################
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs", len(gpus))
tf.debugging.set_log_device_placement(True)


#https://github.com/tensorflow/tensorflow/issues/7072 (karthikeyan19)
#No clue what this does but it fixes CUBLAS errors
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

strategy = tf.distribute.MirroredStrategy()
my_scope = strategy.scope()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
##################
## RUN TRAINING ##
##################

with my_scope:
    tf.random.set_seed(69)
    a = tf.random.uniform((3,2,4))
    b = tf.random.uniform((3,2,4))
    c = tf.math.multiply(a, b)
    print(c.numpy())
