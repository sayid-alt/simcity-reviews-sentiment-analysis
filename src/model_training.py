from sklearn.model_selection import LearningCurveDisplay, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_80, y_train_80)

y_pred_train_80 = rfc.predict(X_train_80)
y_pred_test_20 = rfc.predict(X_test_20)

train_rfc_accuracy_80 = accuracy_score(y_pred_train_80, y_train_80)
test_rfc_accuracy_70 = accuracy_score(y_pred_test_20, y_test_20)

print(f'random forest training accuracy score: {train_rfc_accuracy_80:.3f}')
print(f'random forest testing accuracy score: {test_rfc_accuracy_70:.3f}')

# input_user = input("Input sentence: ")

# preprop = preprocessed_pipeline(input_user)
# tfidf_vector = tfidf.transform([preprop])

# rfc.predict(tfidf_vector)

"""### **7.3.2 `Random Forest` with `TF-IDF` and `70/30` split**"""

rfc.fit(X_train_70, y_train_70)

y_pred_train_70 = rfc.predict(X_train_70)
y_pred_test_30 = rfc.predict(X_test_30)

train_rfc_accuracy_70 = accuracy_score(y_pred_train_70, y_train_70)
test_rfc_accuracy_30 = accuracy_score(y_pred_test_30, y_test_30)

print(f'random forest training accuracy score: {train_rfc_accuracy_70}')
print(f'random forest testing accuracy score: {test_rfc_accuracy_30}')

# # Inference for the Model
# input_user = input("Input sentence: ")

# preprop = preprocessed_pipeline(input_user)
# tfidf_vector = tfidf.transform([preprop])

# rfc.predict(tfidf_vector)

"""### **7.3.3 `LSTM` with `Word2Vec` and `80/20` split**"""

# Define the input shape
input = tf.keras.layers.Input((max_seq_length,), name='input_layer')
embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dims,
                                            weights=[embedding_matrix], trainable=False)(input)

# Add layersΩΩ
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, name="bilstm_layer",
                                                       return_sequences=True))(embedding_layer)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, name="bilstm_layer_2",
                                                       return_sequences=False))(x)
x = tf.keras.layers.Dense(64, activation="relu", name="ff_1")(x)
x = tf.keras.layers.Dense(16, activation='relu', name="ff_2")(x)
outputs = tf.keras.layers.Dense(
    1, activation="sigmoid", name="output_layer")(x)

# Create the model
model = tf.keras.Model(inputs=input, outputs=outputs, name="build_model")

# Print the model summary
model.summary()


class EarlyStopTrainingAtAccuracy(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy):
        super().__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs={}):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if (acc is not None and val_acc is not None) and (acc >= target_accuracy and val_acc >= target_accuracy):
            print(
                f'accuracy has reached the target ({self.target_accuracy}), stop training!')
            self.model.stop_training = True


target_accuracy = 0.921
stop_training_cb = EarlyStopTrainingAtAccuracy(target_accuracy)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='accuracy',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='max',
    min_lr=0.00001,
)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=5e-4, momentum=0.9),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(train_set, epochs=50, validation_data=test_set,
                    batch_size=batch_size, callbacks=[stop_training_cb, reduce_lr])

# Plot Training & Validation Loss
plt.figure(figsize=(14, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
