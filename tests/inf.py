from mrcnn import model as modellib, visualize
from mrcnn.config import Config
import cv2
import matplotlib.pyplot as plt

# Define the configuration for inference
class InferenceConfig(Config):
    NAME = "daps_cfg"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 3  # Background + space-occupied + space-empty
    DETECTION_MIN_CONFIDENCE = 0.5  # Lower threshold if needed

inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config, 
                          model_dir='models')

model.load_weights('models/01_4_20_daps.h5', by_name=True)

test_image_path = 'input/umich/1.jpg'

# Load the image as-is, without any preprocessing, conversuon or modification
image = cv2.imread(test_image_path)  # Load in BGR format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

# Display the input image for reference 
# (to check whether the correct image is being loaded, because it happened so many times that I realized after inference that the model ran on the wrong image)
plt.imshow(image)
plt.axis('off')
plt.show()

# Running detection on the input image
results = model.detect([image], verbose=1)

r = results[0]

class_names = ['BG', 'space-empty', 'space-occupied']

visualize.display_instances(image=image, 
                            boxes=r['rois'], 
                            masks=r['masks'], 
                            class_ids=r['class_ids'], 
                            class_names=class_names, 
                            scores=r['scores'])


colors = {
    'space-empty': (0, 255, 0),  # Blue in BGR
    'space-occupied': (255, 0, 0)  # Red in BGR
}

# Iterating through the detections and modifying the output by adding the labels
for i, box in enumerate(r['rois']):
    y1, x1, y2, x2 = box
    class_id = r['class_ids'][i]
    label = class_names[class_id]
    score = r['scores'][i]
    
    color = colors.get(label, (0, 255, 0))

    #to add bounding boxes
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    #adding the text onto the labelled image
    cv2.putText(image, "{0} {1:.2f}".format(label, score), 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2)

output_path = 'output/umichlot/1_01_4_20_output_image.jpg'
cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
print("Annotated image saved to {}".format(output_path))

print("Detections:")
print("Class IDs:", r['class_ids'])
print("Scores:", r['scores'])
print("Boxes:", r['rois'])
print("Masks shape:", r['masks'].shape)