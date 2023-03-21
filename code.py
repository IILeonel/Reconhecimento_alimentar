# Imports

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Versão do TensorFlow:", tf.__version__)
import keras as K
print("Versão do Keras:", K.__version__)
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix
#from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from loguru import logger
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



## --- Mostra Imagem e faz predição
def Predicao(image_path):
    # test_image = image.load_img(image_path, target_size = (64, 64))
    test_image = tf.keras.utils.load_img(image_path, target_size = (64, 64))
    # test_image = image.img_to_array(test_image)
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    result2 = classifier.predict_step(test_image)
    training_set.class_indices
    print(result)
    print(result2)
    if result[0][0] == 1: prediction = 'Tem Camarão'
    else: prediction = 'Não Tem Camarão'
    # img = image.load_img(image_path)
    img = tf.keras.utils.load_img(image_path)
    img.show()
    print('A imagem ', prediction)

# ------------------------------------------------------------------
# Início Arquitetura da CNN
# ------------------------------------------------------------------
# Inicializando a Rede Neural Convolucional
classifier = Sequential()

# Passo 1 - Primeira Camada de Convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Passo 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adicionando a Segunda Camada de Convolução
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

# Adicionando umaa camada de pooling à saída da camada de convolução anterior
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Passo 3 - Flattening
classifier.add(Flatten())

# Passo 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

'''
# Carregar modelo:load json and create model
json_file = open('CNN_model.json', 'r')
classifier = json_file.read()
json_file.close()
classifier = tf.keras.models.model_from_json(classifier)
# load weights into new model
classifier.load_weights("model.h5")
print("Loaded model from disk")
'''

# Compilando a rede
classifier.compile(optimizer = 'RMSProb', loss = 'binary_crossentropy', metrics=[ 'accuracy', 'mse'])
# Rede Montada

# ------------------------------------------------------------------
# Treinamento da CNN
# ------------------------------------------------------------------

# Pré-Processamento
#-----------------------------
# Criando os objetos train_datagen e validation_datagen com as regras de pré-processamento das imagens


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

# Pré-processamento das imagens de treino e validação
training_set = train_datagen.flow_from_directory('dataset_treino',
                                                 target_size = (64,64),
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 shuffle = True)

validation_set = validation_datagen.flow_from_directory('dataset_validation',
                                                 target_size = (64,64),
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 shuffle = True)



test_set = test_datagen.flow_from_directory('dataset_teste',
                                                 target_size = (64,64),
                                                 batch_size = 64,
                                                 class_mode = 'binary',
                                                 shuffle = True)

# ------------------------------------------------------------------
# Treinamento (esse processo pode levar bastante tempo, dependendo do seu computador)
# ------------------------------------------------------------------
try:
    classifier.fit(training_set,
                             steps_per_epoch = int(11000/64),
                             epochs = 1,
                             validation_data = validation_set,
                             validation_steps = int(4000/64))
except Exception as e:
    logger.exception(f"DEU RUIM {e}")


# ------------------------------------------------------------------

# # Avaliação de Desempenho
# # ------------------------------------------------------------------
# print('--------------- Avaliação de Desempenho ---------------')
# score = classifier.evaluate(training_set, verbose=0)
# print('Treinamento ->',"%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))
# print('-----------------------------------------------')
# print('-----------------------------------------------')
# score = classifier.evaluate(validation_set, verbose=0)
# print('Validação ->',"%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))
# print('-------------------------------------------------------')
# print('-----------------------------------------------')
# score = classifier.evaluate(test_set, verbose=0)
# print('Teste ->',"%s: %.2f%%" % (classifier.metrics_names[1], score[1]*100))
# print('-------------------------------------------------------')

# y_pred = classifier.predict(test_set)
# confusion_matrix_df = pd.DataFrame(confusion_matrix(test_set, y_pred)) #Avaliação de Desempenho 01
#
# plt.subplots(figsize=(12,10))
# sns.heatmap(confusion_matrix_df, annot=True, fmt='g')

#------------------------------------------------------------------
## Salvar modelo: serialize model to JSON
#------------------------------------------------------------------
'''
model_json = classifier.to_json()
with open("CNN_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
'''
#------------------------------------------------------------------

# ------------------------------------------------------------------
# Predição
# ------------------------------------------------------------------



Predicao('teste_5.jpg')
Predicao('teste_1.jpg')
Predicao('teste_2.jpg')
Predicao('teste_12.jpg')
Predicao('teste_13.jpg')
Predicao('teste_14.jpg')
Predicao('teste_15.jpg')
Predicao('teste_16.jpg')

print('='*30)

Predicao('teste_3.jpg') #Camarao
Predicao('teste_7.jpg') #Camarao
Predicao('teste_8.jpg') #Camarao
Predicao('teste_9.jpg') #Camarao
Predicao('teste_10.jpg') #Camarao
Predicao('teste_11.jpg') #Camarao
Predicao('teste_4.jpg') #Camarao
Predicao('teste_6.jpg') #Camarao

'''
------------------------------------------------------------------------------------
Possibilidades de Melhorias Adicionais:
------------------------------------------------------------------------------------
-> Aumentar o número de épocas para 25 para uma aprendizagem mais profunda.
-> Além disso, aumentar o redimensionamento da imagem de 64x64 para 256x256 deve levar a melhores resultados devido à resolução mais alta.
-> Aumentar o tamanho do lote de 32 para 64 também pode levar a melhores resultados.
-> Usar imagens sintéticas rotacionando a imagem principal, técnica conhecida como Dataset Augmentation.
-> Alterar a arquitetura da rede incluindo mais uma camada convolucional.
-> Avaliar outras métricas do modelo e ajustar os hiperparâmetros de acordo.
-> Experimentar outros algoritmos de otimização.
'''

