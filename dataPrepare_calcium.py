#if you don't want to do data augmentation, set data_gen_args as an empty dict.
#data_gen_args = dict()
from data import *

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGenerator = trainGenerator(20,'data/calcium/train','image','label',data_gen_args,save_to_dir = "data/calcium/train/aug")

#you will see 60 transformed images and their masks in data/membrane/train/aug
num_batch = 20
for i,batch in enumerate(myGenerator):
    if(i >= num_batch):
        break

image_arr,mask_arr = geneTrainNpy("data/calcium/train/aug/","data/calcium/train/aug/")
np.save("data/image_arr.npy",image_arr)
np.save("data/mask_arr.npy",mask_arr)