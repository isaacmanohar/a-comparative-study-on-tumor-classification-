import logging, sys
logging.disable(sys.maxsize)
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

# Simulate MRI
model = tf.keras.models.load_model('mri_model.h5')
last_conv_layer_name = None
for layer in reversed(model.layers):
    if hasattr(layer, 'name') and 'conv' in layer.name.lower():
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name:
    inp = tf.keras.Input(shape=(64, 64, 3))
    x = inp
    conv_out = None
    for layer in model.layers:
        try:
            x = layer(x)
            if layer.name == last_conv_layer_name:
                conv_out = x
        except Exception as e:
            print(f"Error in layer {layer.name}: {e}")
            sys.exit(1)
            
    try:
        grad_model = tf.keras.models.Model(inputs=inp, outputs=[conv_out, x])
    except Exception as e:
        print(f"Error in Model: {e}")
        sys.exit(1)
        
    try:
        c_out, preds = grad_model(np.zeros((1, 64, 64, 3), dtype=np.float32))
    except Exception as e:
        print(f"Error in inference: {e}")
        sys.exit(1)
        
    print("SUCCESS")
