import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from preprocessing import *
from sklearn.preprocessing import StandardScaler
import time


def model_fit(model, X, y):
    start_time = time.time()
    X_train, X_test = X
    y_train, y_test = y

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_model_accuracy = accuracy_score(y_pred_train, y_train)
    test_model_accuracy = accuracy_score(y_pred_test, y_test)

    print(f'{type(model).__name__} training accuracy score: {train_model_accuracy:.3f}')
    print(f'{type(model).__name__} testing accuracy score: {test_model_accuracy:.3f}')

    print(f'time consumed: {time.time() - start_time:.3f} sec')

    return model


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
    monitor='val_loss',
    factor=0.2,
    patience=2,
    verbose=1,
    mode='min',
    min_lr=0.001,
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

    # labeling
    df_cleaned['labels'] = df_cleaned['content'].apply(analysis_label)
    print(f'content labels added!')

    # set y variable as a label
    y = df_cleaned['labels']
    print(f'Successfull preprocessed, X shape: {X.shape}, y shape: {y.shape}')

    # get X transformed and tfidf model
    tfidf, X_tfidf = fit_transform_tfidf(X)
    print(f'successfull transform X to tfidf extraction')

    # splitting tfidf transformed d
    X_preproc, y_preproc = split_data(X_tfidf, y)
    print(f'splitted train and test data for rfc model')

    # Train model
    # Train rfc model
    print(f'trainind rfc model for split 80/20')

    rfc_model_80 = model_fit(RandomForestClassifier(random_state=42),
                             X_preproc['80'], y_preproc['80'])

    # train gradient boosting model
    print(f'trainind Gradient Boosting model for split 70/30')
    gb_model_70 = model_fit(HistGradientBoostingClassifier(),
                            X_preproc['70'], y_preproc['70'])

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
