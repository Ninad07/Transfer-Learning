from keras.applications.vgg16 import VGG16
img_row=224
img_col=224
model = VGG16(weights='imagenet', include_top=False, input_shape=(img_row,img_col,3))
for (i,layer) in enumerate(model.layers):
    print (str(i)+" "+layer.__class__.__name__,layer.trainable)
def addTopModel(bottom_model, num_classes):
    """creates the head of the model that will bw placed on top of the bottom layers"""
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(1024,activation='relu')(top_model)
    top_model = Dense(512,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model
    
    model.layers
    
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 2

FC_Head = addTopModel(model, num_classes)
modelnew = Model(inputs=model.input, outputs=FC_Head)
print(modelnew.summary())

from keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'C://Users//Ninad//Desktop//MLOPS//ComputerVision//ml4//train'
validation_data_dir = 'C://Users//Ninad//Desktop//MLOPS//ComputerVision//ml4//test//New Folder'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)

train_batchsize = 16
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_row, img_col),
                                                    batch_size=train_batchsize,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                    target_size=(img_row, img_col),
                                                    batch_size=val_batchsize,
                                                    class_mode='categorical',
                                                    shuffle=False)
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("face_vgg.h5", monitor="val_loss", mode="min", save_best_only = True, verbose=1)
earlystop = EarlyStopping(monitor= 'val_loss', min_delta = 0, patience = 3, verbose = 1, restore_best_weights = True)
callbacks = [earlystop, checkpoint]

modelnew.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001),metrics=['accuracy'])

nb_train_samples=1190
nb_validation_samples=170
epochs=3
batch_size=16

history = modelnew.fit_generator(train_generator,
                                 steps_per_epoch=nb_train_samples // batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=validation_generator,
                                 validation_steps=nb_validation_samples // batch_size)
modelnew.save("Transfer_Learning.h5")


