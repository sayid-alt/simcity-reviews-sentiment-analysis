import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from preprocessing import *


def rfc_model(X, y):
    X_train, X_test = X
    y_train, y_test = y

    rfc = RandomForestClassifier(random_state=42)
    rfc.fit(X_train, y_train)

    y_pred_train = rfc.predict(X_train)
    y_pred_test = rfc.predict(X_test)

    train_rfc_accuracy = accuracy_score(y_pred_train, y_train)
    test_rfc_accuracy = accuracy_score(y_pred_test, y_test)

    print(f'random forest training accuracy score: {train_rfc_accuracy:.3f}')
    print(f'random forest testing accuracy score: {test_rfc_accuracy:.3f}')

    return rfc


def lstm_model(X, y):

    word2vec, params = word2vec_embedding(X)
    X_padded, labels, tokenizer, params = X_y_tokenizing(X, y, params)
    embedding_matrix, params = embed_matrix(tokenizer, word2vec, params)
    train_set, test_set, params = tfds_split(X_padded, labels, params)

    max_seq_length = params['max_seq_length']
    vocab_size = params['vocab_size']
    embedding_dims = params['embedding_dims']
    batch_size = params['batch_size']

    # Define the input shape
    input = tf.keras.layers.Input((max_seq_length,), name='input_layer')

    # embedding layer
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                output_dim=embedding_dims,
                                                weights=[embedding_matrix],
                                                trainable=False)(input)

    # bilstm
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, name="bilstm_layer",
                                                           return_sequences=True))(embedding_layer)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, name="bilstm_layer_2",
                                                           return_sequences=False))(x)

    # full-connected layers
    x = tf.keras.layers.Dense(64, activation="relu", name="ff_1")(x)
    x = tf.keras.layers.Dense(16, activation='relu', name="ff_2")(x)
    outputs = tf.keras.layers.Dense(
        1, activation="sigmoid", name="output_layer")(x)

    # Create the model
    model = tf.keras.Model(inputs=input, outputs=outputs, name="build_model")

    # Print the model summary
    model.summary()

    return model, train_set, test_set, batch_size

###### set callbacks ######


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


if __name__ == "__main__":
    # Load data
    df = loading_dataset()
    df_cleaned = cleaning_data(df)
    print(f'dataset cleaned!')

    # Preprocessing
    X = df_cleaned['content'].apply(preprocessed_pipeline)
    print(
        f'Content has preprocessed,\nan example of processed content: {X.iloc[0]}')

    df_cleaned['labels'] = df_cleaned['content'].apply(analysis_label)
    print(f'content labels added!')

    y = df_cleaned['labels']
    print(f'Successfull preprocessed, X shape: {X.shape}, y shape: {y.shape}')

    tfidf, X_tfidf = fit_transform_tfidf(X)
    print(f'successfull tfidf transforming to X')
    X_rfc, y_rfc = split_data(X_tfidf, y)
    print(f'splitted train and test data for rfc model')

    # Train model
    print(f'trainind rfc model for split 80/20')
    rfc_model_80 = rfc_model(X_rfc['80'], y_rfc['80'])

    print(f'trainind rfc model for split 70/30')
    rfc_model_70 = rfc_model(X_rfc['70'], y_rfc['70'])

    # define lstm model
    lstm_model, train_set, test_set, batch_size = lstm_model(X, y)

    if lstm_model:
        lstm_model.compile(
            optimizer=tf.keras.optimizers.RMSprop(
                learning_rate=5e-4, momentum=0.9),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )

        history = lstm_model.fit(train_set, epochs=50, validation_data=test_set,
                                 batch_size=batch_size, callbacks=[stop_training_cb, reduce_lr])
