# Tube_Contour_Detection

 - **Paper**: [A fully convolutional network for tube contour detection via multi-exposure images (Submitted for Publication)](***)
 - **Dataset**: [Multi-exposure tube contour dataset (METCD)](https://drive.google.com/file/d/1wJ91exa__UEvhM0dzqUviNRASE1Q4JtD/view?usp=sharing)

    Aiming at the problems of difficult and inaccurate extraction of tube contours under complex background, we propose a new tube contour detection method based on the fully convolutional network (FCN). 

### Download Code and Dataset

1. Clone the TubeContourDetection repository
    ```Shell
    git clone https://github.com/chexqi/Tube_Contour_Detection.git
    ```
    We'll call the directory that you cloned TubeContourDetection into `TCD_ROOT`.

2. Download [METCD](https://drive.google.com/file/d/1wJ91exa__UEvhM0dzqUviNRASE1Q4JtD/view?usp=sharing), put it under`TCD_ROOT`
    ```Shell
    $TCD_ROOT/METCD
    ```
    The METCD contains multi-exposure (ME) images of 72 different scenes constructed with tubes, 30 of them are used for FCN training (train set), 10 of them are used for evaluation (validation set), and the rest are used for additional testing (test set).
    
    Each sample of this dataset contains 9 images collected at different exposure times, the corresponding HDR image and tube contour labels with different widths.
    
    ![image](https://github.com/chexqi/Tube_Contour_Detection/blob/master/A_sequence_of_tube_ME_images.jpg)
    
    ![image](https://github.com/chexqi/Tube_Contour_Detection/blob/master/HDR_image_and_labels.jpg)
    
3. Pre-trained model can alse be [downloaded](https://drive.google.com/file/d/1YGyoxAHBpFO6YnNNlwvqitJu_NDmrzHi/view?usp=sharing) directly for validation or testing.

### Experimental environment

    python              3.6.7
    opencv-python       3.4.3.18   
    torch               1.4.0                 
    torchsummary        1.5.1                 
    torchvision         0.5.0                 
    Some other libraries (find what you miss when running the code.)
    
### Preparation for Training, Evaluation and Testing
1. Training
    ```Shell
    $TCD_ROOT python _01TrainMain.py
    ```
    The FCN takes ME images of a static scene as input. Each group includes under-exposure, normal-exposure, and over-exposure images, so as to ensure that the network can obtain the information of tube contours in different dynamic ranges.

2. Validation
    ```Shell
    $TCD_ROOT python _20ValiMain.py
    ```
    Evaluation with `TCD_ROOT/METCD/Val`. We employ three evaluation metrics: 
    
    (1) Mean average precision (mAP), the higher the better.
     
    (2) Maximum F-measure at optimal dataset scale (MF-ODS), the higher the better.
     
    (3) Dilate inaccuracy at optimal dataset scale (DIA-ODS), the lower the better.
    
3. Testing
    ```Shell
    $TCD_ROOT python _30TestMain.py
    ```
    Evaluation with `TCD_ROOT/METCD/Test`. Here are same samples of the tube contour detection results.
    
    ![image](https://github.com/chexqi/Tube_Contour_Detection/blob/master/Tube_contour_detection_results.jpg)

### License

This code and METCD is released under the MIT License (refer to the LICENSE file for details).


### Citing

If you find this code or METCD useful in your research, please consider citing:

    @article{TubeContourDetection_METCD,
        Author = {Xiaoqi Cheng, Junhua Sun, Fuqiang Zhou},
        Title = {A Fully Convolutional Network-Based Tube Contour Detection Method Using Multi-Exposure Images},
        Journal = {Submitted to ****},
        Year = {2021.**}
    }

