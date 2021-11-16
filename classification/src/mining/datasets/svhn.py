import glob
import math
import os
import pickle
import random
import numpy as np
from ..utils import load_indices


class SVHN():
    """process original training data for active learning
    -save original as one pickle
    -read by indices
    -save every block data as one pickle
    """
    # constant var
    data_num = 73257
    class_num = 10

    def __init__(self, data_dir, work_step_dir, initial_ratio=None, query_indices=None, prev_labeled_file=None, prev_unlabeled_file=None):

        self.data_path = data_dir
        # temp path to save processed data
        self.work_step_dir = work_step_dir
        self._all_data_file = "all_data.pickle"
        self.root_dir = os.path.abspath(os.path.join(work_step_dir, "..")) # get parent directory
        self._all_data_path = os.path.join(self.root_dir, self._all_data_file)
        self._test_data_path = os.path.join(self.data_path, "test")

        # data indices
        self.all_indices = list(range(self.data_num))
        self.labeled_indices = []
        self.unlabeled_indices = []

        if not query_indices:
            self.load_dataset_init()
            # generate initial labeled file and initial unlabeled file 
            init_labeled_indices = random.sample(self.all_indices, math.ceil(initial_ratio * self.data_num))
            # save labeled data file and unlabeled data file in work_step_dir
            self.query(init_labeled_indices)
        else:
            # init data indices from previous step dir
            self.labeled_indices = load_indices(prev_labeled_file)
            self.unlabeled_indices = load_indices(prev_unlabeled_file)
            # query and save new labeled data file and new unlabeled data file in work_step_dir
            self.query(query_indices)

    def load_dataset_init(self):
        """load cifar100 training data, indices it and save as one pickle
        return:
            the num of dataset sample
        """
        path = os.path.join(self.data_path, "train")
        
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        inputs = data[b"data"]
        labels = data[b'fine_labels']

        # transform input to 50000*3072
        inputs = np.vstack(inputs)

        # combine to one dict
        all_data = {}
        all_data[b"data"] = inputs
        all_data[b"labels"] = labels
        all_data[b"img_id"] = self.all_indices

        # save in data.pickle
        with open(self._all_data_path, 'wb') as f:
            pickle.dump(all_data, f)


    def query(self, new_indices):
        """add new_data to labeled_data and save
        param:
            new_data_indices: the indices list of new data
        """
        # update indices
        self.labeled_indices = list(self.labeled_indices + new_indices)
        self.unlabeled_indices = list(np.setdiff1d(self.all_indices, self.labeled_indices))

        # save file
        self.save(self.labeled_indices, os.path.join(self.work_step_dir, "labeled_data.pickle"))
        self.save(self.unlabeled_indices, os.path.join(self.work_step_dir, "unlabeled_data.pickle"))
        self.save(new_indices, os.path.join(self.work_step_dir, "new_added_data.pickle"))

    def save(self, data_indices, out_file_name):
        """save data to pickle file according to data indices
        param:
            data_indices: the indices array of data
            out_file_name: the pickle flie in temp floder which contain data
        """
        # read all_data
        with open(self._all_data_path, "rb") as f:
            all_data = pickle.load(f, encoding="bytes")

        # select data according to indices
        selected_data = {}
        selected_data[b"data"] = [all_data[b"data"][i] for i in data_indices]
        selected_data[b"labels"] = [all_data[b"labels"][i] for i in data_indices]
        selected_data[b"img_id"] = data_indices

        # save in data.pickle
        out_path = os.path.join(self.work_step_dir, out_file_name)
        with open(out_path, 'wb') as f:
            pickle.dump(selected_data, f)
