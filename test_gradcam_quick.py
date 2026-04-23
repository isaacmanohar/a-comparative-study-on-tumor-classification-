"""Verify the improved layer selection produces a localized heatmap."""
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('mri_model.h5')

# New selection logic: find conv layer with spatial >= 8x8
target_layer = None
fallback = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        if fallback is None:
            fallback = layer.name
        out_shape = layer.output.shape
        if len(out_shape) >= 3 and out_shape[1] is not None and out_shape[1] >= 8:
            target_layer = layer.name
            break

if target_layer is None:
    target_layer = fallback

print(f"Selected layer: {target_layer}")
print(f"(Fallback was: {fallback})")

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(target_layer).output, model.output]
)

img = np.random.rand(1, 128, 128, 3).astype(np.float32)

with tf.GradientTape() as tape:
    conv_out, preds = grad_model(img)
    tape.watch(conv_out)
    pred_idx = tf.argmax(preds[0]).numpy()
    loss = preds[:, pred_idx]

grads = tape.gradient(loss, conv_out)
pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
heatmap = conv_out[0] @ pooled[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0)
max_val = tf.reduce_max(heatmap)
if max_val > 0:
    heatmap = heatmap / max_val
heatmap = heatmap.numpy()

print(f"Heatmap shape: {heatmap.shape}")
print(f"Heatmap min/max: {heatmap.min():.4f} / {heatmap.max():.4f}")
print(f"Heatmap > 0.5 fraction: {(heatmap > 0.5).sum() / heatmap.size:.2%}")
print(f"Heatmap > 0.25 fraction: {(heatmap > 0.25).sum() / heatmap.size:.2%}")
print(f"Heatmap == 0 fraction: {(heatmap == 0).sum() / heatmap.size:.2%}")
