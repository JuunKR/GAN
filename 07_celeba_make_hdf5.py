import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import zipfile


total_images = 20000

hdf5_file = '_data/celeba.h5py'

with h5py.File(hdf5_file, 'w') as hf:

    count = 0

    with zipfile.ZipFile('_data/celeba.zip', 'r') as zf:
      for i in zf.namelist():
        if (i[-4:] == '.jpg'):
          # extract image
          ofile = zf.extract(i)
          img = imageio.imread(ofile)
          os.remove(ofile)

          # add image data to HDF5 file with new name
          hf.create_dataset('img_align_celeba/'+str(count)+'.jpg', data=img, compression="gzip", compression_opts=9)
          
          count = count + 1
          if (count%1000 == 0):
            print("images done .. ", count)
            pass
            
          # stop when total_images reached
          if (count == total_images):
            break
          pass

        pass
      pass