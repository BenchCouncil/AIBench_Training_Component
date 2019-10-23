import time
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

if __name__ == '__main__':
    query_category_input = Input(shape=(1,), dtype='int32', name='category')
    user_sex_input = Input(shape=(1,), dtype='int32', name='sex')
    user_age_input = Input(shape=(1,), dtype='int32', name='age')
    user_power_input = Input(shape=(1,), dtype='int32', name='power')

    query_category_layer = Embedding(output_dim=16, input_dim=1000000, input_length=1)(query_category_input)
    user_sex_layer = Embedding(output_dim=2, input_dim=2, input_length=1)(user_sex_input)
    user_age_layer = Embedding(output_dim=8, input_dim=150, input_length=1)(user_age_input)
    user_power_layer = Embedding(output_dim=2, input_dim=7, input_length=1)(user_power_input)

    embedding_concat = keras.layers.concatenate(
        [query_category_layer, user_sex_layer, user_age_layer, user_power_layer])

    x = LSTM(32)(embedding_concat)
    x = Dense(64, activation='relu', name='Relu_1')(x)
    x = Dense(3, activation='relu', name='Relu_2')(x)

    sub_model = Model(inputs=[query_category_input, user_sex_input, user_age_input, user_power_input], outputs=[x])
    sub_model.predict([[2345], [0], [28], [6]])
    feature_input = Input(shape=(3,), dtype='float', name='feature_input')

    x = keras.layers.dot([feature_input, sub_model.output], axes=1)
    x = Dense(1, activation='sigmoid', name='Sigmoid')(x)
    model = Model(inputs=[query_category_input, user_sex_input, user_age_input, user_power_input, feature_input],
                  outputs=[x])
    query_category_data = np.random.randint(1000000, size=(1000, 1))
    user_sex_data = np.random.randint(2, size=(1000, 1))
    user_age_data = np.random.randint(150, size=(1000, 1))
    user_power_data = np.random.randint(7, size=(1000, 1))
    feature_data = np.random.random(size=(1000, 3))

    labels = np.random.random(size=(1000, 1))

    print("Start training\n")
    start = time.time()
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit([query_category_data, user_sex_data, user_age_data, user_power_data, feature_data], [labels], epochs=200,
              batch_size=32)
    end = time.time()
    print("Training time: {} seconds\n".format(end - start))

    test_query_category_data = np.random.randint(1000000, size=(1000000, 1))
    test_user_sex_data = np.random.randint(2, size=(1000000, 1))
    test_user_age_data = np.random.randint(150, size=(1000000, 1))
    test_user_power_data = np.random.randint(7, size=(1000000, 1))
    test_feature_data = np.random.random(size=(1000000, 3))

    # model.predict([[2345], [0], [28], [6], [[20.4, 23.4, 29.8]]])
    start_time=time.time()
    a = model.predict([test_query_category_data, test_user_sex_data, test_user_age_data, test_user_power_data,
                         test_feature_data])
    print("test_time:",time.time()-start_time)
