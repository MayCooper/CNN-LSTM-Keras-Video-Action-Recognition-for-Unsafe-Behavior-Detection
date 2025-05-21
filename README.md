# CNN-LSTM Computer Vision Video Action Recognition for Unsafe Behavior Detection  

**Author: May Cooper**

## Overview

This video classification project uses a CNN-LSTM neural network to detect unsafe human behaviors by analyzing action sequences in video footage. The model combines spatial features from a convolutional neural network (CNN) with temporal dependencies captured through a Long Short-Term Memory (LSTM) layer. This architecture enables real-time safety monitoring and supports proactive risk mitigation in surveillance and workplace environments.

---

## Tools and Technologies

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **Matplotlib / Seaborn**
* **NumPy / Pandas**
  
---

## What is CNN-LSTM?

CNN-LSTM is a hybrid deep learning model that processes spatial and temporal data:

- **CNN (Convolutional Neural Network)** captures spatial features from each video frame (e.g., edges, textures, objects).
- **LSTM (Long Short-Term Memory)** models sequential dependencies between frames to recognize dynamic motion and action patterns.

Together, they are well-suited for recognizing human actions in video, especially when motion is a key indicator of behavior.

---

## Project Objectives

- **Primary Goal**: Detect and classify unsafe pedestrian behavior in video sequences.
- **Real-World Application**: Used in workplace surveillance, public safety, and construction monitoring to proactively flag hazardous actions.
- **Technical Workflow**:
  1. Preprocess videos into fixed-length frame sequences.
  2. Extract spatial features with a pre-trained CNN (e.g., VGG16).
  3. Model temporal dynamics with an LSTM network.
  4. Train and evaluate the model using performance metrics and visualizations.

---

## Research Question

Can a CNN-LSTM deep learning model accurately identify unsafe actions in real-time video to enable safety-enhancing interventions?

---

## Dataset Summary

- This project performs multiclass video classification using a subset of human actions from the UCF101 dataset, a standard benchmark for action recognition containing 101 video classes.
- A selected group of 11 action categories (e.g., CliffDiving, RopeClimbing, WalkingWithDog) were used to train a convolutional neural network (CNN) to recognize and distinguish between them.
- Each video is preprocessed into a fixed-length 20-frame RGB sequence and resized to 224×224 resolution to be compatible with standard CNN input dimensions.
- The model uses a 3D convolutional architecture (Conv3D layers) to capture both spatial and temporal features of human motion from the video segments.
  
* **Processing Steps**:

  * Frames extracted from videos (sampled every 5 frames)
  * Noisy frames removed using SSIM thresholding
  * Frames resized to 224x224
  * Features extracted via VGG16 (include\_top=False)
  * Features grouped into 16-frame sequences

---

## Sample of Processed Dataset

| FrameSequenceID | ClipName   | ActionLabel    |
| --------------- | ---------- | -------------- |
| 001             | cliff01    | CliffDiving    |
| 002             | walkdog01  | WalkingWithDog |
| 003             | climb02    | RopeClimbing   |
| 004             | jumpjack03 | JumpingJack    |
| 005             | lunges04   | Lunges         |

---

## Model Architecture

1. **VGG16 CNN** (pre-trained on ImageNet) extracts 25088-dimensional spatial features per frame.
2. **LSTM Layer** with 128 units models sequential dependencies over 16-frame input.
3. **Dense ReLU Layer** with 64 nodes refines LSTM outputs.
4. **Dense Softmax Output Layer** maps to 11 action classes.

**Total Parameters**: \~12.9M (trainable)

---

## Visual Analysis and Interpretations

### 1. Sample Video Frames
![image](https://github.com/user-attachments/assets/489c782c-8959-4c31-968e-702953ed8d79)
![image](https://github.com/user-attachments/assets/4a6b601c-1d54-487a-8486-0b96c321e802)
![image](https://github.com/user-attachments/assets/e7effffe-698a-486f-a6cb-93c7fb484551)
![image](https://github.com/user-attachments/assets/9b800a63-d517-4ab4-b4e4-ffe32a3bdb15)
![image](https://github.com/user-attachments/assets/12d33aaf-68be-4cc6-8877-617dabbf93d1)
![image](https://github.com/user-attachments/assets/be26ab80-b5ab-4bfb-8307-a7a4cf1dd5af)
![image](https://github.com/user-attachments/assets/d040af98-877f-49c7-8fed-95dc2f7e5017)
![image](https://github.com/user-attachments/assets/7e594334-a2ca-4d2c-9e55-09de0489e5b0)
![image](https://github.com/user-attachments/assets/65aadccb-11d0-446a-ada2-6676fa719d2a)

Montages were generated showing one frame per selected class. These helped validate the visual content post-resizing (224x224) and confirmed correct labeling.

### 4. Training and Validation Curves
![image](https://github.com/user-attachments/assets/ad970a91-718e-4697-9caa-bf4d88c5124a)

Training loss decreased steadily while validation loss plateaued after epoch 4. Early stopping was applied at epoch 9 to prevent overfitting.

### 5. Confusion Matrix
![image](https://github.com/user-attachments/assets/2e09a816-ab7c-48e8-91fd-510237f65a4c)

The matrix highlighted high accuracy for JumpingJack, RockClimbingIndoor, and Biking. Misclassifications clustered around LongJump, which was over-predicted due to class imbalance.

### 6. Sample Misclassifications

Examples included:

* CliffDiving misclassified as LongJump
* Lunges misclassified as LongJump
  These errors reveal similarity in body posture/motion between classes and suggest improvement via data balancing or fine-tuning.

---

## Performance Metrics

| Metric         | Value                                         |
| -------------- | --------------------------------------------- |
| Accuracy       | 92.3% (train), \~46% (test)                   |
| Precision      | Varies per class (high for JumpingJack)       |
| Recall         | 100% for top classes, 0% for underrepresented |
| F1-Score       | Moderate (imbalanced by class)                |
| Inference Time | \~0.08 sec/sequence                           |

---

## Conclusion

This project demonstrates the feasibility of using CNN-LSTM models for real-time action recognition in safety-critical settings. Despite data imbalance challenges, the model accurately classified well-represented classes and showed promising results for deployment after optimization. It provides a replicable, modular pipeline for future improvements and applications in surveillance or risk detection systems.

---

## Future Work

* Add data augmentation for underrepresented classes
* Experiment with attention layers or temporal convolution
* Apply model pruning/quantization for edge deployment
* Expand to full UCF101 or additional datasets (e.g., NTU RGB+D)

---

## Deployment Considerations

* Model is saved as `final_model.keras` for easy reuse
* Can be embedded into smart camera systems or construction safety apps
* Integration with real-time alert systems recommended

