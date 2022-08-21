# Object-tracking
1
Abstract—Video analysis is one of the fastest-growing and most 
important areas of research in today's society. It is the process of 
evaluating video automatically to detect various actions. Industry 
automation is rapidly increasing. The automation industry's 
efficiency is improved by object recognition and tracking. The 
research focuses mostly on object detection and tracking in a 
video. Moving Object detection is the first low-level important task 
for any video application. Detection of moving objects is a 
challenging task. Tracking is required in higher-level applications 
that require the location and shape of an object in every frame. It 
is done with the help of video processing and Python. Frame 
differencing is used for detecting the movement of the object. The 
code is simulated in Python.
Index Terms— Object Detection, Object tracking, background 
subtraction,
I. INTRODUCTION
HE demand for intelligent surveillance camera systems has 
surged as public awareness of security has grown.
Object detection has benefited from technological 
advancements and the usage of computer technology. Real-time 
visual tracking of object and visual attributes of an image is a 
difficult task with numerous possible applications. It is a most 
important topic in computer vision. It serves as a foundation for 
many technologies in developing real world computer vision 
applications: human tracking, autonomous vehicles, video 
surveillance, vehicles traffic control, visual tracking system for 
security purpose, industry automation etc.
The development of this Project contains three parts. The 
first is detecting an object based on one or more of its features. 
When motion is detected, the second step is to track the detected 
object. Third is video processing the first and second stages 
using python.
The timelines for the above tasks are as follows:
Object Detection: It involves object recognition in a video 
frame. The available method for object detection are as follows 
Optical flow, frame differencing, background subtraction etc.
Object Tracking: The fundamental goal is to track an object in 
a series of frames. Both Kernel-based tracking and contourbased tracking can be used to implement this step. 
Video processing: It is nothing but performing operations on a 
video frame by frame. Frames are nothing more than a single 
instance of the video at a specific point in time. Even in a single 
second, we can have numerous frames. Frames are similar to 
images in that they can be treated as such. As a result, all actions 
that we can conduct on images may also be performed on 
frames.
At present, the frame subtraction method, background
subtraction method and optical flow method are the most 
widely utilized methods in moving object detection [5]. The
presence of moving objects is determined by using frame 
subtraction approach, which is difference between two 
consecutive images. The optical flow method is for calculating 
the image flow field according to the distribution characteristic 
of the optical flow of the image and performing the clustering 
process [6]. Background subtraction method is a way to remove 
the background from the current image. This extracts the
animated foreground from the static background, but it is very 
sensitive to changes in the environment and poor antiinterference ability. In this paper, the motion detection is used 
based on background subtraction method. It is also a Gaussian 
Mixture-based Background/Foreground Segmentation 
Algorithm. It provides better adaptability to varying scenes due 
illumination changes etc.
II. OBJECT DETECTION
The detection of moving objects is the basic step for others
Video analysis. All tracking methods require an object
detection mechanism for all frames or objects first appeared in 
A MOVING OBJECT DETECTION AND 
TRACKING IN A VIDEO BY USING 
COMPUTER VISION TECHNIQUES
Vikram Narre, ECE Graduate Student, University of Florida
T
2
the video. It distinguishes between moving and immovable 
background objects [10]. This concentrates on higher-order 
processing. It also cuts down on computation time. Shadow 
object segmentation becomes tough and problematic as ambient 
circumstances such as lighting change. Using information from 
a single frame is a frequent method for object detection. To limit 
the frequency of false detections, several object detection 
systems leverage the temporal information generated from a 
sequence of frames [11]. Frame differencing, which shows 
regions that vary dynamically in consecutive frames, is one way 
to get this temporal information. Given the images object 
regions, the tracker must conduct object correspondence from 
one frame to the next in order to generate the tracks.
Background subtraction, temporal difference, and statistical 
approaches are the three moving object identification methods 
discussed in this section.
Figure 1: Framework of Moving Object Detection System [13]
Primarily, In the object detection we need to distinguish
foreground objects from still background. To do this, we can 
construct a foreground pixel map at each frame using a 
combination of techniques and low-level image post-processing 
approaches. The related regions in the foreground map are then 
grouped to extract individual object attributes like bounding 
box, area, and perimeter.
Foreground detection major aim is to recognize foreground 
objects from the stationary background. In every object
detection process first thing is to detect the foreground objects.
This concentrates attention on higher processing levels such as 
tracking, categorization, and behavior interpretation, while also 
reducing computing time because only pixels corresponding to 
foreground objects must be processed [15].
The creation of the background scene is the first step. The 
background scene is modelled through a variety of techniques.
The system's background scene-related components are 
isolated, and their coupling with other modules is maintained to 
a minimal, allowing the entire detection system to work with 
any of the background models [16]. Detecting the foreground 
pixels using the background model and the current image from 
video is the next step in the detection method. This pixel-level 
detection technique is dependent on the background model in 
use, and it is used to update the background model in order for 
it to accommodate to dynamic scene changes [14].
In addition, the identified foreground pixel map contains 
noise due to camera noise or environmental factors. To
eliminate noise in the foreground pixels, pixel-level postprocessing methods are used. After we get the filtered 
foreground pixels, we use a connected component labeling 
algorithm to find connected regions and calculate the bounding 
rectangles of the objects. Due to flaws in the foreground 
segmentation method, the labeled regions may encompass 
nearby but discontinuous regions. As a result, in the regionlevel post-processing stage, several relatively small regions 
created by environmental noise are removed. Using the 
foreground pixel map, several object attributes such as area, 
bounding box, and perimeter of the regions corresponding to 
objects are extracted from the current image in the final step of 
the detection process.
Figure 2: Foreground detection from stationary background.
Post pixel processing: Noise is generated in the output of 
foreground detection.It is generally influenced by a variety of 
noise components. To alleviate the problem of noise, more 
pixel-level processing is required. Noise in foreground 
detection is caused by various factors, including:
Camera Noise:The image acquisition components of the camera 
cause camera noise. This is the noise produced by the image 
acquisition components of the camera. The intensity of a pixel 
that corresponds to an edge between two distinct colored 
objects in the scene may be set to one of the object's color in 
one frame and the other's color in the next frame, resulting in 
noise[11].
Background Colored Object Noise: The object's color may 
match the reference background's color. With the use of a 
reference background, detecting foreground pixels is 
challenging [11].
Reflection Noise: The light source causes reflection noise. 
Some components of the backdrop scene reflect light when a 
light source changes from one place to another [11].
To eliminate noise induced by the things listed above, we can 
apply a low pass filter and morphological processes such as 
3
erosion and dilation to the foreground pixel map [16]. The goal 
of these operations is to remove noisy foreground pixels that do 
not match to actual foreground regions, as well as noisy 
background pixels that are actually foreground pixels near and 
inside object regions. Low pass filters are used to blur images 
and reduce noise. Blurring is employed in pre-processing tasks 
including removing little details from an image before 
extracting large objects and bridging small gaps in lines or 
curves. For pixel level post processing, a Gaussian low pass 
filter is used [12]. By calculating weighted averages in a filter 
co-efficient, a Gaussian filter smooths an image. The input 
signal is modified by convolution with a Gaussian function in a 
Gaussian filter.
Figure 3: Pixel level post-processing of moving objects.
The filtered foreground pixels are organized into connected 
areas after detecting foreground regions and conducting postprocessing techniques to remove noisy regions. The bounding 
boxes of these regions are calculated after locating individual 
regions that relate to objects.
III. OBJECT TRACKING
In human motion analysis, object tracking is a critical topic. 
This is a problem with higher-level computer vision. Tracking
is the process of matching identified foreground objects 
between consecutive frames using various object characteristics 
such as motion, velocity, color, and texture. Object tracking in 
a surveillance system is the process of monitoring an object 
over time by determining its position in each frame of the video. 
It may also fill in the image region that is occupied by the object 
at any given time [17]. The objects are represented using form 
or appearance models in the tracking approach. The type of 
motion is limited by the model chosen to represent object shape. 
If an item is represented as a point, for example, the sole option 
is to utilize a translational model. Parametric motion models 
such as affine or projective transformations are useful when a 
geometric shape representation such as an ellipse is employed 
for the item [16]. The motion of stiff objects in the scene can be 
approximated using these representations. The most descriptive 
representation for a non-rigid object is a silhouette or contour, 
and both parametric and nonparametric models can be 
employed to specify their motion. 
In this paper, we are discussing kernel tracking. The object's
shape and appearance are required by the kernel. In this method, 
any feature of an object, such as a rectangular template or an 
elliptic shape with an associated histogram, is utilized to 
monitor it as a kernel. After determining the kernel's mobility 
between successive frames, the object can be tracked.
Figure 4: Block diagram for object tracking [18].
IV. EVALUATION
In this project, I used the Euclidean Distance tracker for 
tracking the object. The object tracking algorithm is kernal
tracking because it relies on the euclidean distance between (1) 
existing object centroids (i.e., items the centroid tracker has 
already observed) and (2) new object centroids between 
consecutive frames in a video. This algorithm is a multi-step 
process. In each frame, the tracking algorithm assumes that we 
are passing in a set of the bounding box (x, y)-coordinates for 
each detected object. Using a beginning set of object detections 
(such as an input set of bounding box coordinates). Creating a 
distinct ID for each of the first detections. Then, while 
maintaining the assignment of unique IDs, track each of the 
objects as they move around frames in a video. Object tracking 
also allows us to assign a unique ID to each tracked object, 
allowing us to count the number of unique objects in a video.
This unique ID’s future used for counting the objects.
The Euclidean distance formula is as follows:
Figure 5: Euclidean Distance formula
4
The algorithm which I used for my implementation is as 
follows:
By using this algorithm, the tracking of the object is easy. 
Compared with the other models like yolo4 the computational 
process of the video is very slow. In this model, the video 
processing is from a static camera and tracking of the object
detection is depending also depends on the camera stability.
The bounding box for each object is created for detection.
Masking is used for clearing the background and making 
grayscale. Then contour is used for detecting the shape of the 
object. Using the object shape the rectangular bounding boxes 
are created for detecting the objects. This model detection of an 
object in a given region has comparatively better accuracy than 
the other models.
Outputs for the moving object detection using the static camera 
in a video.
Figure 6:Intial frame of the frame.
Figure 7: Conture frame of the object detection in the video
before cleaning.
Figure 8: Conture frame of the object detection in the video
after cleaning.
Figure 9: Final object detection in a video.
V. Related work 
Several authors have attempted to design a robust method for 
real-time tracking for a video sequence with a static camera. In
this method, I used background subtraction with masking and 
contours for detecting the object and tracking. The 
computational cost of object tracking is high. Optical flow, 
parametric model motion, and matching blocks. and other ways 
for tracking objects aim to reduce the computational cost.
There are different models which are also used to detect and 
track the object like Single Shot Detector (SSD), YOLO (You 
Only Look Once). The YOLO model in this model there are 
different versions of models that exist. Here we mainly focus 
on YOLOv4. YOLOv4 is a one-stage object detection model 
that improves on YOLOv3 by incorporating many literaturebased methods and modules. Compared with the model that 
used in the project the YOLOv4 requires a high computation 
process. As I compared with the model in the project the 
YOLOv4 has Comparatively low recall and more localization 
error and also struggles to detect small objects.
A single-shot detector like YOLO takes only one shot to 
detect numerous objects present in an image using a multi-box. 
It has a substantially faster object detecting system with good 
accuracy. When compared with the single-shot detector with 
this model the single-shot detector has high accuracy than this 
model.
Some of the cons in this project are as follows: whenever 
there is disturbance in video then this model detects the other 
5
objects that are not required for our purposes. It is sensitive to 
environmental noise.
Pros for this project are as follows: It requires object detection 
phase once (i.e., when the object is initially detected), It will 
be extremely fast – considerably faster than running the object 
detector itself. It is able to pick objects it has lost in between 
frames. It is robust occlusion. able to handle when the tracked 
object “disappears” or moves outside the boundaries of the 
video frame.
VI. CONCLUSION
In this paper, I build an object detection and tracking 
model using OpenCV and python. I have seen about image 
processing and object representation, object detection, feature
selection of tracking, object tracking. I also learned about the 
background subtraction, optical flow method and frame 
differencing. I also learned about OpenCV and different object 
detection and tracking models.
REFERENCES
[1] Prerna Dewan, Rakesh Kumar, “Detection of the object in motion using
improvised background subtraction algorithm,” proceeding of IEEE 
conference on trends in electronics and informatics, pp. 651-656, 2017.
[2] Olivier Barnich and Marc Van Droogenbroeck, Member, IEEE,”ViBe: A 
Universal Background Subtraction Algorithm for Video Sequence”, 
IEEE Transaction on Image Processing Vol.20, No.6, June 2011.
[3] In Su Kim, Hong Seok Choi, Kwang Moo Yi, Jin Young Choi, and Seong 
G. Kong. Intelligent Visual Surveillance - A Survey. International 
Journal of Control, Automation, and Systems (2010) 8(5):926-939.
[4] Li, Y.; Wang, J.; Xing, T. TAD16K: An enhanced benchmark for 
autonomous driving. In Proceedings of the 2017 IEEE International 
Conference on Image Processing (ICIP), Beijing, China, 17–20 
September 2017; pp. 2344–2348.
[5] Amedome Min-Dianey Kodjo, Yang Jinhua: “Real-time Moving Object 
Tracking in Video”. International Journal of Soft Computing and 
Engineering (IJSCE) ISSN: 2231-2307, Volume-2, Issue-3, July 2012
[6] Pande R.P., Mishra N.D., Gulhane S. and,” Detection of moving object 
with help of motion detection alarm system” journal of signal and image 
processing, May 03, 2012
[7] Ming-Yu Shih, Yao-Jen Chang, Bwo-Chau Fu, and Ching-Chun Huang,” 
Motion-based Background Modeling for Moving Object Detection on 
Moving Platforms” IEEE 2007 
[8] Kenji Iwata, Yutaka Satoh, Ikushi Yoda, and Katsuhiko Sakaue “Hybrid 
Camera Surveillance System by Using Stereo Omni-directional System 
and Robust Human Detection” First Pacific Rim Symposium, PSIVT 
2006 Hsinchu, Taiwan, December 2006
[9] N. Paragios, and R. Deriche.. Geodesic active contours and level sets for 
the detection and tracking of moving objects. IEEE Trans. Patt. Analy. 
Mach. Intell. 22, 3, 266–280, 2000.
[10] Nikita Ghode, P.H. Bhagat: Motion Detection Using Continuous Frame 
Difference and Contour Based Tracking. Proceedings of the Third 
International Conference on Trends in Electronics and Informatics 
(ICOEI 2019) IEEE Xplore Part Number: CFP19J32-ART; ISBN: 978-
1-5386-9439-8
[11] N. Paragios, and R. Deriche.. Geodesic active contours and level sets for 
the detection and tracking of moving objects. IEEE Trans. Patt. Analy. 
Mach. Intell. 22, 3, 266–280, 2000
[12] Elgammal, A.,Duraiswami, R.,Harwood, D., Anddavis, L. 2002. 
Background and foreground modeling using nonparametric kernel 
density estimation for visual surveillance. Proceedings of IEEE 90, 7, 
1151–1163
[13] 
https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researc
hgate.net%2Ffigure%2FFigure-31-Flow-diagram-of-moving-objectdetection_fig10_308694385&psig=AOvVaw1IJFt_1cTMaqTSSOoiSc7
N&ust=1639327707779000&source=images&cd=vfe&ved=2ahUKEwj
W3ufemdz0AhVMAN8KHUuIChMQr4kDegUIARC7AQ
[14] S. Zhu, and A. Yuille. Region competition: unifying snakes, region 
growing, and bayes/mdl for multiband image segmentation. IEEE Trans. 
Patt. Analy. Mach. Intell. 18, 9, 884–900, 1996.
[15] M. Kass, A. Witkin, and D. Terzopoulos. Snakes: active contour models. 
Int. J. Comput. Vision 1, 321–332, 1988.
[16] Kinjal A Joshi, Darshak G. Thakore: A Survey on Moving Object 
Detection and Tracking in Video Surveillance System. International 
Journal of Soft Computing and Engineering (IJSCE) ISSN: 2231-2307, 
Volume-2, Issue-3, July 2012
[17] Yilmaz, A., Javed, O., and Shah, M. 2006. Object tracking: A survey. 
ACM Comput. Surv. 38, 4, Article 13,December 2006
[18]
https://www.google.com/imgres?imgurl=https%3A%2F%2Fwww.resea
rchgate.net%2Fprofile%2FHabib-Mohammed2%2Fpublication%2F325735814%2Ffigure%2Ffig2%2FAS%3A63698
5098137605%401528880490967%2FBlock-Diagram-for-Objecttrackingsystem.png&imgrefurl=https%3A%2F%2Fwww.researchgate.net%2Ffi
gure%2FBlock-Diagram-for-Object-trackingsystem_fig2_325735814&tbnid=8DdJDaKHALQaCM&vet=12ahUKE
wj9s7O4gN30AhWLGt8KHfsxDg0QMygBegUIARCuAQ..i&docid=b
JCV98HWQUF48M&w=537&h=555&itg=1&q=tracking%20object%2
0block%20diagram&ved=2ahUKEwj9s7O4gN30AhWLGt8KHfsxDg0
QMygBegUIARCuAQ
