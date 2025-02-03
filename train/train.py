import os
import xml.etree.ElementTree
from numpy import zeros, asarray
import mrcnn.utils
import mrcnn.config
import mrcnn.model


class dapsDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "space-empty")
        self.add_class("dataset", 2, "space-occupied")

        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        for filename in os.listdir(images_dir):
            image_id = filename
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id.replace('.jpg', '.xml')

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)
        root = tree.getroot()

        boxes = []
        class_names = []
        for obj in root.findall('.//object'):
            name = obj.find('name').text
            class_names.append(name)

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height, class_names

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']

        # Extract bounding boxes and class labels
        boxes, w, h, class_names = self.extract_boxes(path)

        # Initialize masks
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1

            class_ids.append(self.class_names.index(class_names[i]))

        return masks, asarray(class_ids, dtype='int32')


class dapsConfig(mrcnn.config.Config):
    NAME = "daps_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + space-empty + space-occupied
    STEPS_PER_EPOCH = 5


# Load dataset and initialize Mask R-CNN
train_set = dapsDataset()
train_set.load_dataset(dataset_dir='data/train', is_train=True)
train_set.prepare()

valid_dataset = dapsDataset()
valid_dataset.load_dataset(dataset_dir='data/valid', is_train=False)
valid_dataset.prepare()

daps_config = dapsConfig()

model = mrcnn.model.MaskRCNN(mode='training',
                             model_dir='train',
                             config=daps_config)

model.load_weights(filepath='mask_rcnn_coco.h5', by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(train_dataset=train_set, val_dataset=valid_dataset,
            learning_rate=daps_config.LEARNING_RATE,
            layers='heads',
            epochs=20)

model_path = 'train/mask_rcnn_coco.h5'
model.keras_model.save_weights(model_path)
