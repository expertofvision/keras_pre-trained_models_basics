from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.nasnet import preprocess_input, decode_predictions
import numpy as np


def nasnetmobile_classify():
    print('I am nasnetmobile - Classify...............................')

    model = NASNetMobile(weights='imagenet')

    img_path = 'banana.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print('Predicted:', decode_predictions(preds, top=3)[0])
    # Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]


    print('Finish nasnetmobile processing - Classify..................')

def nasnetmobile_features():
    print('I am nasnetmobile - Features...............................')

    model = NASNetMobile(weights='imagenet', include_top=False)

    img_path = 'banana.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = model.predict(x)
    print('Shape', features.shape)
    print('Features.........................')
    print(features)

    print('Finish nasnetmobile processing - Features..................')
