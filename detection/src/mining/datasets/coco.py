import json
import numpy as np
import cv2
import numpy as np
import json
import io

def coco_load(path, filter_unlabeled=True, config=None):
    coco_ann = json.load(open(path))
    image_set = []
    meta = {'info':coco_ann['info'], 'licenses':coco_ann['licenses'],'categories':coco_ann['categories']}
    images = coco_ann['images']
    instances = {}
    image_info = {}
    for index in range(len(images)):
        image_info[images[index]['id']] = images[index]
        instances[images[index]['id']] = []
    annotations = coco_ann['annotations']
    for index in range(len(annotations)):
        # filter pseudo label from last step
        if annotations[index].get("is_pseudo_label", False):
            continue
        instances[annotations[index]['image_id']].append(annotations[index])
    for index in range(len(images)):
        # filter images with empty bbox
        if filter_unlabeled and len(instances[images[index]['id']]) == 0:
            continue
        tmp = {}
        tmp['info'] = image_info[images[index]['id']]
        tmp['instances'] = instances[images[index]['id']]
        image_set.append(tmp)
    return image_set, meta


def coco_combine(image_sets):
    combined_set = []
    for image_set in image_sets:
        combined_set += image_set
    return combined_set
        

def coco_save(image_set, meta, save_path, config=None):
    coco_anno = {}
    coco_anno.update(meta)
    images = []
    annotations = []
    for index in range(len(image_set)):
        images.append(image_set[index]['info'])
        annotations += image_set[index]['instances']
    coco_anno['images'] = images
    coco_anno['annotations'] = annotations

    file = open(save_path, 'w')
    file.write(json.dumps(coco_anno))
    file.close()