# CNN-LSTM Video Action Recognition with UCF101

**Author: May Cooper**

## Overview

This project applies a CNN-LSTM deep learning model to classify human actions in video clips, using a curated subset of the UCF101 dataset. The model is designed to identify pedestrian behaviors that may be deemed unsafe in real-world environments like construction zones. By combining spatial features from a pre-trained VGG16 convolutional network with temporal dependencies captured by an LSTM, the system learns to recognize and differentiate between actions such as CliffDiving, JumpingJack, and WalkingWithDog. The model supports multi-class classification and is optimized for motion-based activity recognition.

---

## Purpose and Research Question

**Research Question**: Can action recognition help detect unsafe pedestrian behavior, such as running or cliff-diving, near construction zones to prevent accidents?

This project explores whether a CNN-LSTM model can classify potentially hazardous human actions using video footage, aiding real-time safety monitoring systems.

**Objectives:**

* Filter relevant pedestrian-like actions from UCF101.
* Extract visual and temporal features.
* Train a model to classify actions.
* Enable real-time applicability for safety use cases.

---

## What is CNN-LSTM?

**CNN-LSTM** is a hybrid model architecture:

* **CNN (Convolutional Neural Network)**: Extracts spatial features from individual frames using VGG16.
* **LSTM (Long Short-Term Memory)**: Captures motion trends and temporal patterns across sequences of frames.

This architecture is ideal for tasks where both static visual appearance and dynamic motion are essential.

---

## Dataset Summary

* **Source**: UCF101 human action recognition dataset
* **Selected 11 Action Classes**: BalanceBeam, Biking, CliffDiving, HandstandWalking, JumpingJack, JumpRope, LongJump, Lunges, RockClimbingIndoor, RopeClimbing, WalkingWithDog
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

## Tools and Technologies

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **Matplotlib / Seaborn**
* **NumPy / Pandas**

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

