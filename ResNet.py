# Imports
import os
from keras.preprocessing.image import ImageDataGenerator
import preprocess as prep
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import pandas as pd


# Function definitions

def train_model(model,datapath_training, datapath_val, optimizer_type='adam', loss_type='categorical_crossentropy', metrics_type='accuracy'):
    # compile
    model.compile(optimizer=optimizer_type, loss=loss_type, metrics=[metrics_type])
    
    # creating image generators
    train_datagen = ImageDataGenerator(fill_mode = 'constant', cval = '0.0', rescale=1./255)
    val_datagen = ImageDataGenerator(fill_mode = 'constant', cval = '0.0', rescale=1./255)
    
    
    train_generator = train_datagen.flow_from_directory(
        datapath_training,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        seed= 223)

    validation_generator = val_datagen.flow_from_directory(
        datapath_val,
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        seed=223)
    
    #training
    hitstory = model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=20)
    
    return history

def make_model(): 
    
    base_model = ResNet50(include_top=False, weights='imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    predictions = Dense(40, activation='softmax')(x)
    return Model(inputs = base_model.input, outputs = predictions)

def make_test_gen(datapath_test):
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
    datapath_test,
    target_size=(224,224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=1,
    seed=223)

    
def plot_acc(history_object):
    plt.plot(history_object.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.savefig('mlp_acc_64.png')
    plt.clf()
    

def main():
    
    model = make_model()
    datapath_training = os.getcwd() + "/Data/preprocessed_imgs/training_set"
    datapath_val = os.getcwd() + "/Data/preprocessed_imgs/validation_set"
    
    history = train_model(model, datapath_training, datapath_val)
    plot_acc(history)
    
    datapath_test = os.getcwd() + "/Data/preprocessed_imgs/test_set"
    test_generator = make_test_gen(datapath_test)
    
    score = model.evaluate_generator(test_generator)
    with open('output_score.txt', 'w') as f:
        print('Filename:', 'score', file=f)
    
    test_generator.reset()
    predictions = model.predict_generator(test_generator, verbose=1)
    
    with open('output_predictions.txt', 'w') as f:
        print('Filename:' 'score', file = f)
   
    predicted_class_indices = np.argmax(pred, axis = 1)
    
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    predictions = [labels for k in predicted_class_indices]
    
    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    results.to_csv("results.csv", index=False)
    
    
    
    

if __name__ == "__main__":
            main()