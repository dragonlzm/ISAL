import scipy.io
import numpy as np
import pickle

#mat = scipy.io.loadmat('train_32x32.mat')
mat = scipy.io.loadmat('test_32x32.mat')

image_list = []
label_list = []

# the size of the train set and the test set are 73257 and 26032
for i in range(mat['X'].shape[-1]):	
	this_img = mat['X'][:,:,:,i]
	this_img = this_img.reshape(-1)
	image_list.append(this_img)
image_list = np.array(image_list)

for i in range(mat['X'].shape[-1]):	
	this_label = mat['y'][i][0]
	label_list.append(this_label)

converted_result = {}
converted_result[b"data"] = image_list
converted_result[b'fine_labels'] = label_list

# save in data.pickle
#with open('train', 'wb') as f:
with open('test', 'wb') as f:
    pickle.dump(converted_result, f)