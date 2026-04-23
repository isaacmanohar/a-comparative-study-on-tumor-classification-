import tensorflow as tf
import numpy as np

# Load
model_us = tf.keras.models.load_model('breast_model.h5')

last_conv_layer_name = None
for layer in reversed(model_us.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

if last_conv_layer_name:
    print("Rebuilding model graph safely...")
    inp = tf.keras.Input(shape=(128, 128, 3))
    x = inp
    conv_out = None
    for layer in model_us.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_out = x
            
    try:
        grad_model = tf.keras.models.Model(inp, [conv_out, x])
    except Exception as e:
        print(f"FAILED: {e}")
        import sys
        sys.exit(1)
    
    img_batch = np.zeros((1, 128, 128, 3), dtype=np.float32)
    with tf.GradientTape() as tape:
        c_out, preds = grad_model(img_batch)
        loss = preds[:, 0]
        
    grads = tape.gradient(loss, c_out)
    print("Grads shape:", grads.shape)
    print("SUCCESS")
