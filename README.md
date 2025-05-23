# CNN-LSTM Computer Vision Video Action Recognition for Unsafe Behavior Detection

**Author: May Cooper**

## Overview

This end-to-end computer vision project implements a CNN-LSTM neural architecture to detect potentially unsafe pedestrian behaviors in video streams. The system combines spatial understanding from a Convolutional Neural Network (CNN) with temporal pattern recognition from a Long Short-Term Memory (LSTM) network. This architecture enables real-time surveillance applications aimed at preventing accidents and enhancing safety compliance in environments like construction zones and public areas.

---

## Tools and Technologies

* **Python**
* **TensorFlow / Keras**
* **OpenCV**
* **Matplotlib / Seaborn**
* **NumPy / Pandas**

---

## What is CNN-LSTM?

CNN-LSTM is a hybrid deep learning architecture tailored for video analysis:

* **CNN** extracts meaningful visual features from each frame, identifying spatial patterns such as edges, textures, and objects.
* **LSTM** models the temporal progression of these features, enabling recognition of motion-based activities over time.

This fusion of spatial and sequential learning is ideal for detecting human actions where behavior unfolds across multiple frames.

---

## Research Question

Can action recognition help detect unsafe pedestrian behavior, such as running or cliff-diving, near construction zones to prevent accidents?

Or more formally:

How accurately can a 2D CNN + LSTM architecture classify potentially unsafe pedestrian behaviors relevant to real-world safety monitoring scenarios?

The system aims to support real-time detection of high-risk actions in surveillance contexts, particularly within hazardous or regulated environments like construction sites. These areas are prone to accidents due to heavy machinery, uneven surfaces, and restricted zones, making proactive behavior detection essential.

---

## Project Objectives

### 1. Data Filtering and Preparation

* Identify and select video actions that reflect pedestrian-like or potentially unsafe activities.
* Clean and standardize frame sequences to remove noise and enable efficient processing.

### 2. Feature Extraction and Sequence Construction

* Use a pre-trained 2D CNN (e.g., VGG16) to extract spatial features from individual frames.
* Organize frames into fixed-length sequences (e.g., 16-frame windows) to encode motion over time.

### 3. Model Training and Evaluation

* Train an LSTM-based sequence model to learn temporal behavior patterns.
* Split the data into training, validation, and test sets.
* Evaluate classification accuracy using precision, recall, F1-score, and confusion matrices.

### 4. Deployment and Real-Time Readiness

* Prepare the model pipeline for real-time integration in safety systems.
* Explore model compression strategies (e.g., pruning, quantization) to enable edge deployment and lower-latency inference.

---

## Applications

* **Construction Site Monitoring**: Automatically detect high-risk actions near heavy equipment or in restricted areas.
* **Smart Surveillance Systems**: Flag potentially dangerous behavior in public safety camera feeds.
* **Workplace Safety Compliance**: Identify deviations from safety protocols in industrial or logistics environments.

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

# Exploratory Data Analysis and Preparation

## Data Overview

To support the research question on unsafe pedestrian behavior near construction zones, we selected action categories from a broader video dataset that exhibit relevant movement patterns. The EDA revealed a fairly balanced distribution across categories, supporting unbiased training and evaluation.

Video Category Distribution

To ensure the dataset supports meaningful classification of unsafe pedestrian behavior, only a subset of videos from the larger dataset was selected. The chart below shows the number of videos per category after applying the filtering process:
![image](https://github.com/user-attachments/assets/3c1d00a2-6ab4-4c9e-ab1b-2c9fe68854dc)

This relatively balanced distribution across categories helps ensure that no single class dominates the training process, promoting generalization and robust learning.

* **Average Video Duration**: Categories like "JumpRope" and "RockClimbingIndoor" had longer durations, indicating more motion complexity.
* This variability guided filtering, balancing, and model-ready preprocessing.

![image](https://github.com/user-attachments/assets/54c2d6df-b8cc-4ab8-9670-f87d1b0d4fb6)

## Train-Validation-Test Split

* **Categories Selected**: Only videos resembling pedestrian behaviors were retained.
* **Split Ratios**:

  * Training Set: 70%
  * Validation Set: 15%
  * Testing Set: 15%

![image](https://github.com/user-attachments/assets/a0e65586-1ad6-472b-ae4b-fc76e5edcc4b)
![image](https://github.com/user-attachments/assets/224ca82f-7f33-4210-8294-6f90d8eb4aab)

**Justification**:

* Ample training data ensures feature learning.
* A validation set helps tune hyperparameters (e.g., learning rate, batch size).
* A reserved test set provides unbiased final evaluation.

## Frame Extraction (Sampling Rate = 5)

To reduce redundancy and maintain motion cues:

* Frames sampled at 1 out of every 5.
* Reduces dataset size by 80%, speeds up processing, and retains temporal structure.
* Helps avoid excessive frame similarity in slow or repetitive scenes.
![image](https://github.com/user-attachments/assets/fafe2043-4617-412a-85f0-f7fe1756bccf)

## Frame Filtering (SSIM-Based)

To ensure quality:

* Frames too similar to a blank reference were discarded.
* SSIM threshold: 0.3
* **Total Noisy Frames Removed**: 187
* Categories most affected: *HandstandWalking*, *Lunges*

This improved data quality for subsequent model training.

## Frame Resizing and Visualization

* All frames resized to **224×224** to match VGG16 input requirements.
* Visual inspection confirmed preserved visual clarity and category consistency.

## Feature Extraction (VGG16)

* Each frame passed through pre-trained **VGG16** with `include_top=False`.
* Output feature maps: shape **(7,7,512)** → flattened to **(25088)**
* Output saved as `.npy` files for efficient loading during training.
![image](https://github.com/user-attachments/assets/96fe6b38-a8dc-459a-9d98-a033d93bf00a)

## Sequence Formation

* Grouped 16 consecutive frames into single sequences.
![image](https://github.com/user-attachments/assets/94eb1852-e520-4b30-a1cb-3cbb08d74d95)
* Each sequence: shape **(16, 25088)**
* Categories varied in valid sequence counts due to duration and noise filtering.
* Categories like *RockClimbingIndoor* had high counts; *Lunges* had none.
![image](https://github.com/user-attachments/assets/39ce9ba5-cf41-480a-bd0b-308284225d6d)


### Sample Video Frames
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

## Processed Dataset Summary

* `data/frames`: Resized, filtered images
* `data/features`: VGG16 feature vectors and sequences
* `final_processed_dataset.zip`: Compressed version of the dataset

---

## Justification for Each Data Step

1. **Train-Validation-Test Split**: Promotes generalization and performance tracking.
2. **Frame Sampling**: Maintains motion cues while optimizing processing.
3. **Noisy Frame Removal**: Prevents low-quality data from degrading learning.
4. **Resizing**: Ensures model compatibility and input uniformity.
5. **Feature Extraction**: Reduces computation and applies transfer learning.
6. **Sequence Preparation**: Enables LSTM to learn temporal relationships.

These structured steps formed a reproducible pipeline for video-based behavior recognition.

---

# Network Architecture Summary

## Model Configuration

### Input Layer

* Shape: **(16, 25088)** — a sequence of 16 extracted frame vectors.

### LSTM Layer

* Units: **128**
* Learns temporal dependencies.
* Parameters: **12,911,104**

### Dense Hidden Layer

* Units: **64**
* Activation: ReLU
* Parameters: **8,256**

### Output Layer

* Units: **11** (one for each action category)
* Activation: Softmax
* Parameters: **715**

**Total Parameters**: **12,920,075** — all trainable

![image](https://github.com/user-attachments/assets/3c1b063c-ce07-4d89-ac87-9b7f7df9f8aa)

This compact and efficient design is well-suited for real-time unsafe action detection.

### Training and Validation Curves
![image](https://github.com/user-attachments/assets/ad970a91-718e-4697-9caa-bf4d88c5124a)

Training loss decreased steadily while validation loss plateaued after epoch 4. Early stopping was applied at epoch 9 to prevent overfitting.

### Confusion Matrix
![image](https://github.com/user-attachments/assets/2e09a816-ab7c-48e8-91fd-510237f65a4c)

The matrix highlighted high accuracy for JumpingJack, RockClimbingIndoor, and Biking. Misclassifications clustered around LongJump, which was over-predicted due to class imbalance.

### Sample Misclassifications

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
# Neural Network Architecture and Justification

## Chosen Architecture: 2D CNN (VGG16) + LSTM

This project uses a hybrid deep learning architecture that combines a 2D CNN with an LSTM for effective spatial-temporal modeling in video action recognition. This approach offers a practical balance between performance and efficiency, especially for real-time safety monitoring scenarios.

### Why 2D CNN + LSTM?

* **2D CNN (VGG16)**: Efficiently extracts spatial features such as edges, shapes, and objects from individual frames. Using a pre-trained VGG16 model on ImageNet accelerates training and reduces the need for a large custom dataset.
* **LSTM**: Models temporal dependencies between frames, enabling the system to understand sequences of motion (e.g., jump rope or cliff diving). The LSTM's gating mechanisms help preserve long-term information and mitigate vanishing gradients.
* **Efficiency and Modularity**: Compared to 3D CNNs or Transformers, this combination is lightweight, modular, and well-suited for scalable real-time applications.

### Advantages in Safety Contexts

* **Spatial-Temporal Decoupling**: Separates the tasks of visual interpretation (handled by CNN) and motion sequence learning (handled by LSTM).
* **Real-Time Capable**: Lightweight enough to run on edge devices with techniques like pruning and quantization.
* **Proven Effectiveness**: Widely adopted in surveillance, autonomous systems, and human activity recognition tasks.

---

## Workflow Summary

### 1. Data Preparation

* Filter the video dataset to retain only relevant action categories.
* Conduct EDA to visualize class distributions and identify noise.
* Use SSIM to eliminate low-information frames.

### 2. Frame Preprocessing

* Resize all video frames to 224×224 to match VGG16 input size.
* Standardize image format and perform visual inspection.

### 3. Feature Extraction

* Use the truncated VGG16 model (without dense layers) to convert each frame into a feature vector.
* Save feature maps for efficient sequence assembly.

### 4. Sequence Creation

* Group frame-level features into 16-frame sequences.
* Store sequences as 3D arrays for LSTM consumption.

### 5. Model Training

* Train the CNN-LSTM network using categorical cross-entropy and Adam optimizer.
* Monitor validation loss and adjust training parameters to prevent overfitting.

### 6. Evaluation

* Use accuracy, precision, recall, F1-score, and confusion matrices to assess performance.
* Analyze misclassifications to identify areas for refinement.

### 7. Deployment

* Optimize model via pruning and quantization.
* Integrate into surveillance systems to generate real-time alerts for unsafe behavior.

---

## Expected Outcomes

1. **Trained Classifier**: A robust model distinguishing between safe and unsafe actions.
2. **Reproducible Pipeline**: Documented steps for data processing, feature extraction, and model training.
3. **Deployment Blueprint**: Guidelines for real-world integration and edge device performance.
4. **Improvement Directions**: Recommendations for boosting performance using additional data, alternate architectures (e.g., ResNet, MobileNet), or advanced models like Transformers.

---

This 2D CNN + LSTM approach offers a strong foundation for intelligent video monitoring. Its blend of spatial detail and temporal learning enables actionable insights in safety-critical environments while remaining efficient enough for real-time deployment.

---

## Potential Impact

This research addresses safety challenges in construction environments by facilitating proactive identification and monitoring of potentially hazardous behaviors. Implementing such a system makes it possible to reduce the likelihood of accidents, ensure adherence to safety regulations, and increase the overall safety climate for workers.

The system’s real-world applications include generating real-time alerts for supervisors, triggering automated safety mechanisms, and enabling autonomous vehicles to respond to erratic pedestrian behavior.

Beyond the construction sector, this methodology holds promise for other high-risk areas, such as crowded public venues and industrial sites. Its adaptable design allows for broader applications, supporting efforts to maintain safety and mitigate risks across diverse environments.

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

