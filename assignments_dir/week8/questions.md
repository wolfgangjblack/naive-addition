# Summary Questions

## 1)  Algo Understanding
How does transfer learning work?
    - Transfer learning is a technique in which we generalized a pre-trained neural network to a new task. For example, we can take a model of some architecture trained on ImageNet which has 1000 classes, chop off its top layer (which is the soft max output and usually some global averaging used to flatten the final conv layer) and attach a new global averaging with a new fully connected layer of n nodes. The n nodes correspond to the number of labels for our specific case. We would then retrain with our limited data only the output/global pooling layers, or any number of layers within the original NN. This would be transfer learning to generalize to some new purpose. 
    
When to use transfer learning?
    - Really transfer learning can be used in most instances. Its great when we want to save time and resources to generalize to a new problem when we may not have loads of data. Most transfer learning occurs within a same problem space, so in my example above we took a nn trained on imagenet data and generalized to n classes, which correspond to some new image dataset. However, research is ongoing to use transfer learning across spaces (So nlp transfer learning for music, ect). 
    
    
## 2) Interview Readiness
When training a Convolution Neural Network in the parameters what do each of the letters mean, for example NHWC?

| Parameter  | Tensor | Meaning |
| ------------- | ------------- | ------------- |
| N | N/A | Batch Size |
| C | Input | Number of Channels |
| H | Input | Height |
| W | Input | Width |
| K | Output | Number of Channels (can also be C) |
| P | Output | Height (ofter derived) |
| Q | Output | Width (often dervied) |
| R | Filter | Height | 
| S | Filter | Width |
| U | Filter | Vertical Stride |
| V | Filter | Horizontal Stride |
| PadH | Filter | Vertical Input Padding |
| PadW | Filter | Horizontal Input Padding |
| DilH | Filter | Veritcal Dilation |
| DilW | Filter | Horizontal Dilation |

## 3) Interview Readiness 2
How does an SSD (single shot multi box detector) object detection model work?
 - High detection accuracy in SSD is achieved by using multiple boxes or filters with different sizes, and aspect ratio for object detection. It also applies these filters to multiple feature maps from the later stages of a network. This helps perform detection at multiple scales.
 - At prediction time, the network generates scores for the presense of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally the network combines predictions from the multiple feature maps with different resolutions to naturally handle objects of various sizes. 
 - in summary: it uses multiple boxes and feature maps to detect objects of various sizes and classes in a single pass through the NN. Typically Yolo is a favorite algorithm for this

## 4) Interview Readiness the third
What is Intersection over Union and why do we use Intersection over Union?
    - Intersection over Union is a term used to describe the extent of overlap of two boxes. The greater the region of overlap, the greater the IOU. This is mainly used in object detection when attempting to locate objects optimally in an image. 
    - intersection is calculated as: $I = ( min(x_{2,a}, x_{2,b}) - max(x_{1,a}, x_{1,b}) ) * ( min(y_{2,a}, y_{2,b}) - max(y_{1,a}, y_{1,b}) )$
        - where x and y are coordinates in cartesian space, and subscripts a, b, 1, and 2 denote bounding box a, bounding box b, the first point (or min coordinate of the box) along an axis, and the second point (or max coordinate) along an axis respectively. 
    - Union is calculated as: $U = A_{a} + A_{b} - I$
        - where A is the area of the box and I is the previously calculated intersection
    - Given these equations, IOU can be calculated as $IOU = \frac{I}{U}$

