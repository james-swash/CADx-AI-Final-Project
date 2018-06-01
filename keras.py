import keras

(train_X,train_Y), (test_X,test_Y) = keras.datasets.fashion_mnist.load_data()
print(train_X.shape)