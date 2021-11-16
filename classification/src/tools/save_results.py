"""This file contain useful functions to save or read results
"""
import pickle
import json


def save_results_to_json(mining_method, cur_data_num, al_result, json_file_path):
    """save active learning results to json file
    param:
        mining_method: string name
        cur_data_num: current step labeled data number
        al_result: one active learning result value
        json_file_path: saved json file path
    """
    data_dict = {}
    data_dict["cur_data_num"] = cur_data_num
    data_dict["al_result"] = al_result
    data_dict["mining_method"] = mining_method
    json_str = json.dumps(data_dict)
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_str)
    

def save_predictions_to_pkl(preds_list, id_list, pkl_file_path):
    """save predictions results to pickle file
    param:
        preds_list: predictions list after testing on trained model (data_num *(class_num))
        id_list: image id list
        pkl_file_path: saved file path
    """
    save_data = {}
    save_data[b"pred"] = preds_list
    save_data[b"id"] = id_list

    # save in data.pickle
    with open(pkl_file_path, 'wb') as f:
        pickle.dump(save_data, f)


def read_predictions_from_pkl(pkl_file_path):
    """read predictions results from pkl file
    param:
        pkl_file_path: saved pkl file path
    return:
        pred_list: predictions numpy array (data_num * class_num)
        id_list: image data id numpy array (data_num * 1)
    """
    # read data
    with open(pkl_file_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    return data[b"pred"], data[b"id"]
