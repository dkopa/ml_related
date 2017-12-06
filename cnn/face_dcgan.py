from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input, Lambda, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K

import matplotlib.pyplot as plt
from keras.utils.io_utils import HDF5Matrix

import sys

import numpy as np

class DCGAN():
    def __init__(self):
        self.img_rows = 112
        self.img_cols = 96
        self.channels = 1

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        noise_shape = (100,)
        noise = Input(shape=noise_shape)
        
        fc1 = Dense(128 * 7 * 6, activation="relu")(noise)        
        rs1 = Reshape((7, 6, 128))(fc1)
        bn1 = BatchNormalization()(rs1)
        us1 = UpSampling2D()(bn1)
        
        cn1 = Conv2D(128, kernel_size=3, padding="same")(us1)
        act1 = Activation("relu")(cn1)
        bn2 = BatchNormalization()(act1) 
        us2 = UpSampling2D()(bn2)
        
        cn2 = Conv2D(64, kernel_size=3, padding="same")(us2)
        act2 = Activation("relu")(cn2)
        bn3 = BatchNormalization()(act2)
        us3 = UpSampling2D()(bn3)
        
        cn3 = Conv2D(32, kernel_size=3, padding="same")(us3)
        act3 = Activation("relu")(cn3)
        bn4 = BatchNormalization()(act3)
        us4 = UpSampling2D()(bn4)
        
        cnF = Conv2D(1, kernel_size=3, padding="same")(us4)
        op_img = Activation("tanh")(cnF)
        
        model = Model(inputs=noise, outputs=op_img)        
        model.summary()
        
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)
        # img_shape = (112, 96, 1)
        img = Input(shape=img_shape)

        c1 = Conv2D(32, kernel_size=3, strides=1, padding="same")(img)
        act1 = LeakyReLU(alpha=0.2)(c1)
        mp1 = MaxPooling2D(pool_size=(2, 2))(act1)
        do1 = Dropout(0.25)(mp1)
        
        c2 = Conv2D(64, kernel_size=3, strides=1, padding="same")(do1)
        act2 = LeakyReLU(alpha=0.2)(c2)
        mp2 = MaxPooling2D(pool_size=(2, 2))(act2)
        do2 = Dropout(0.25)(mp2)
        
        bn1 = BatchNormalization()(do2)
        c3 = Conv2D(128, kernel_size=3, strides=1, padding="same")(bn1)
        act3 = LeakyReLU(alpha=0.2)(c3)
        mp3 = MaxPooling2D(pool_size=(2, 2))(act3)
        do3 = Dropout(0.25)(mp3)
        
        bn2 = BatchNormalization()(do3)
        c4 = Conv2D(256, kernel_size=3, strides=1, padding="same")(bn2)
        act4 = LeakyReLU(alpha=0.2)(c4)
        mp4 = MaxPooling2D(pool_size=(2, 2))(act4)
        do4 = Dropout(0.25)(mp4)
        
        bn3 = BatchNormalization()(do4)
        c5 = Conv2D(256, kernel_size=3, strides=1, padding="same")(bn3)
        act5 = LeakyReLU(alpha=0.2)(c5)
        mp5 = MaxPooling2D(pool_size=(2, 2))(act5)
        do5 = Dropout(0.25)(mp5)
        
        flat = Flatten()(do5)
        fc1 = Dense(32, activation='sigmoid')(flat)        
        fc2 = Dense(1, activation='sigmoid')(fc1)
        
        model = Model(inputs=img, outputs=fc2)        
        model.summary()

        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128):
        half_batch = int(batch_size / 2)
        numIterPerSet = 200
        numSampToLoadOnce = half_batch * numIterPerSet
        numTrSamp = 461800
        # numTrSamp = 10000        
        
        # Start training    
        for j in range(epochs):
            currIter = 0
            st = 0
            ed = st + numSampToLoadOnce
            
            while st < numTrSamp:
                print('== Global Epoch: '+ str(j) + ', [' + str(st) + ', ' + str(ed) + '] ===')
                if ed>numTrSamp:
                    ed = numTrSamp
                
                if(ed-st < half_batch):
                    st = ed-1
                    continue
                
                # Load face data
                X_train = HDF5Matrix('/raid5/hasnat/dbs/Casia/DB_tr.h5', 'X', start=st, end=ed)
                # Preprocess data
                X_train = (np.asarray(X_train, dtype='float32') - 127.5) * 0.0078125                                                       
                
                print('Image loaded and preprocessed !')
                          
                for k in range(np.int(X_train.shape[0]/half_batch)):
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    
                    # Select a random half batch of images
                    imgs = X_train[k*half_batch:(k+1)*half_batch]
        
                    # Sample noise and generate a half batch of new images
                    noise = np.random.normal(0, 1, (half_batch, 100))
                    gen_imgs = self.generator.predict(noise)
        
                    # Train the discriminator (real classified as ones and generated as zeros)
                    d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                    d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
                    # ---------------------
                    #  Train Generator
                    # ---------------------
        
                    noise = np.random.normal(0, 1, (batch_size, 100))
        
                    # Train the generator (wants discriminator to mistake images as real)
                    g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))
        
                    # Plot the progress
                    print ("Epoch: %d, iter: %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (j, currIter+k, d_loss[0], 100*d_loss[1], g_loss))
                # update indices
                st = ed
                ed = st + numSampToLoadOnce
                
                # If at save interval => save generated image samples
                self.save_imgs(j, currIter)
                self.save_model(j, currIter)
                currIter = currIter+k
                
                del X_train    
                
    def save_model(self, epoch, citer):
        saveModelPath = '/raid5/hasnat/Keras-GAN/dcgan/saved_model/'
        
        gen_model = self.generator
        # serialize model to JSON
        model_json = gen_model.to_json()
        with open(saveModelPath + 'Gen_epoch_' + str(epoch) + '_iter_' + str(citer) + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        gen_model.save_weights(saveModelPath + 'Gen_epoch_' + str(epoch) + '_iter_' + str(citer) + '.h5')
        
        dis_model = self.discriminator
        # serialize model to JSON
        model_json = dis_model.to_json()
        with open(saveModelPath + 'Dis_epoch_' + str(epoch) + '_iter_' + str(citer) + '.json', "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        dis_model.save_weights(saveModelPath + 'Dis_epoch_' + str(epoch) + '_iter_' + str(citer) + '.h5')
        
        print("Saved model to disk")
            
    def save_imgs(self, epoch, citer):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = np.uint8((gen_imgs/0.0078125) + 127.5)

        fig, axs = plt.subplots(r, c)
        #fig.suptitle("DCGAN: Generated digits", fontsize=12)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("/raid5/hasnat/Keras-GAN/dcgan/images/face_syn_%d_%d.png" % (epoch, citer))
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=5, batch_size=100)
