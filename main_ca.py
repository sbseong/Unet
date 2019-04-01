from model import *
from data_ca import *
import sys
from scipy.io import loadmat
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

myGene = trainGenerator(2,'data/calcium/train','image','label',data_gen_args,save_to_dir = 'data/check')


model = unet()
model_checkpoint = ModelCheckpoint('unet_calcium.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=1000,epochs=3,callbacks=[model_checkpoint])

testGene = testGenerator("data/calcium/test")
results = model.predict_generator(testGene,22,verbose=1)
saveResult("data/calcium/test",results)
