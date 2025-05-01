import tensorflow as tf
import keras
import numpy as np

# Custom Focal Loss function
def focal_loss(alpha, gamma):
    return tf.keras.losses.BinaryFocalCrossentropy(alpha=alpha, gamma=gamma)

# Search Space
alpha_values = [0.25, 0.5]
gamma_values = [1.0, 2.0, 5.0]

best_alpha, best_gamma = None, None
best_val_loss = np.inf

for alpha in alpha_values:
    for gamma in gamma_values:
        print(f"Testing alpha={alpha}, gamma={gamma}")

        model = keras.models.Sequential([...])  # Define your model here
        model.compile(optimizer='adam', loss=focal_loss(alpha, gamma), metrics=['accuracy'])

        history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

        val_loss = min(history.history['val_loss'])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_alpha, best_gamma = alpha, gamma

print(f"Best parameters: alpha={best_alpha}, gamma={best_gamma}")
