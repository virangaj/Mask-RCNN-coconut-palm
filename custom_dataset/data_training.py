import os
import xml.etree
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class cropsDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "coconut")
        self.add_class("dataset", 2, "palm")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if is_train and int(image_id) >= 100:
                continue

            if not is_train and int(image_id) < 100:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            
            if(box[4] == 'coconut'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('coconut'))
            if(box[4] == 'palm'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('palm'))
               
        return masks, asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//object'):
  
            name = box.find('name').text
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class CropConfig(mrcnn.config.Config):
    NAME = "crops_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.001
    NUM_CLASSES = 3
    EPOCH = 10
    STEPS_PER_EPOCH = 10

# Train
train_dataset = cropsDataset()
train_dataset.load_dataset(dataset_dir='data', is_train=True)
train_dataset.prepare()

# Validation
validation_dataset = cropsDataset()
validation_dataset.load_dataset(dataset_dir='data', is_train=False)
validation_dataset.prepare()

# Model Configuration
crop_config = CropConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./', 
                             config=crop_config)

model.load_weights(filepath='mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_dataset, 
            val_dataset=validation_dataset, 
            learning_rate=crop_config.LEARNING_RATE, 
            epochs=crop_config.EPOCH, 
            layers='heads')

model_path = 'crops_mask_rcnn_trained.h5'
model.keras_model.save_weights(model_path)
