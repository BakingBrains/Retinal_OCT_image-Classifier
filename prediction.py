from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import image

model = load_model('Glucoma_classifier.h5')

img_add = 'C:/Users/Prince_Shaks/Desktop/WorkStation/Exciting/Retinal_OCT_images/OCT2017/test/DME/DME-9583225-1.jpeg'
img = image.load_img(img_add, target_size=(150,150))

img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_batch)
print(prediction)