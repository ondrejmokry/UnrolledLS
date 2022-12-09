import tensorflow as tf
import tensorflow_mri as tfmri
from data_loader import MRI_dataset
import matplotlib.pyplot as plt
from model import CP_model
import os
from datetime import datetime

batch_size=1


cartesian=False

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 40GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=10e3)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


with tf.device('/device:GPU:0'):
    dataset_obj=MRI_dataset()

    dataset, kspace,w=dataset_obj.load_data(cartesian_reco=cartesian,batch_size=batch_size)

    el=dataset.get_single_element()

    model=CP_model(kspace,w=w,cartesian_reco=cartesian)
    solution=model.call(el[0])

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(el[0].shape[1:], batch_size=batch_size,dtype=tf.complex64))
    model.add(CP_model(kspace,w=tf.sqrt(w),cartesian_reco=cartesian))

    
    plt.imshow(tf.abs(solution[0,0,1,:,:]))
    plt.show()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='mean_absolute_error', 
              metrics=[tf.keras.metrics.MeanAbsoluteError()])


    
    plt.imshow(tf.abs(solution[0,0,1,:,:]))
    plt.show()    

    #logs = "./logs/" + "1"

    #tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,profile_batch=1)


    a=1
    #model.fit(dataset,epochs=1,batch_size=1)
    
a=1
