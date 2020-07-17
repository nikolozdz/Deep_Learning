from tensorflow import keras
import numpy
model = keras.models.load_model('cifar10.h5')

for i in range(4):
    predictedValue=model.predict(X_test[[i],:])
    predictClasses=model.predict_classes(X_test[[i],:])
    realVal=y_test[[i],:]
    print("Real Value for index: {} Image {}".format(str(i+i),str(numpy.argmax(actual_value))))
