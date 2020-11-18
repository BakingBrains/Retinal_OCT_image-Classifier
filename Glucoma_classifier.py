from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import os


main_dir = 'C:/Users/Prince_Shaks/Desktop/WorkStation/Exciting/Retinal_OCT_images/OCT2017'
train_dir = os.path.join(main_dir, 'train')
val_dir = os.path.join(main_dir, 'val')
test_dir = os.path.join(main_dir, 'test')

print('Training Data:')
print(len(os.listdir(os.path.join(train_dir, 'CNV'))))
print(len(os.listdir(os.path.join(train_dir, 'DME'))))
print(len(os.listdir(os.path.join(train_dir, 'DRUSEN'))))
print(len(os.listdir(os.path.join(train_dir, 'NORMAL'))))

training_datagen = ImageDataGenerator(rescale=1/255,
                                      height_shift_range=0.2,
                                      rotation_range=40,
                                      width_shift_range=0.2,
                                      shear_range=0.2,
                                      horizontal_flip=0.2,
                                      zoom_range=0.2
                                      )

validation_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)

training_generator = training_datagen.flow_from_directory(
                        train_dir,
                        target_size=(150,150),
                        batch_size=50,
                        class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
                        val_dir,
                        target_size=(150,150),
                        batch_size=50,
                        class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
                        test_dir,
                        target_size=(150,150),
                        batch_size=20,
                        class_mode='categorical'
)


model = Sequential()

model.add(Conv2D(64, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
#model.add(Dropout(0.4))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
#model.add(Dropout(0.4))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.summary()

model.compile(Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['acc'])

model.fit(training_generator,
          epochs = 10,
          validation_data=validation_generator)

test_loss, test_acc = model.evaluate(test_generator)
print('Test Loss:{}, Test Accuracy:{}'.format(test_loss,test_acc))

model.save('Glucoma_classifier.h5')