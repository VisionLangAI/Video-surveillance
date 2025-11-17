[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17630997.svg)](https://doi.org/10.5281/zenodo.17630997)

# Criminal Activities for Security Surveillance by KeyFrames Detection and Advanced Inception Techniques  

## Overview  
This project presents a Deep Learning-based framework for the automatic classification of criminal activities in security surveillance videos using Key-Frame Extraction and the Advanced Inception v4 model.  
The research focuses on detecting abnormal activities — Abuse, Arson, Assault, and Arrest — by processing video sequences efficiently through representative frame selection and robust feature extraction.

---

## Research Workflow  

### 1. Dataset Collection  
A publicly available dataset was used for training and validation.  

- **Dataset URL:** https://www.kaggle.com/datasets/umarmominn/human-abnormal-behaviour-detection 
- **Categories:** Abuse, Arson, Assault, Arrest  
- **Total Videos:** 200 (50 per class)  
- **Split:** 70% training / 30% testing  

---

### 2. Methodology  

####  Key-Frame Extraction  
- Extracts the most informative frames from each video.  
- Removes redundant frames to reduce computation.  
- Preserves temporal cues essential for activity recognition.

#### Data Preprocessing  
- Image resizing, normalization  
- Histogram Equalization to improve brightness/contrast consistency  

####  Model Training  
- **Inception v4** (enhanced CNN with factorized convolutions + residual connections)  
- Compared with **Inception v3** to demonstrate improvement  
- Training performed on extracted frames  

#### 📊 Evaluation Metrics  
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix

---

##  Experimental Results  

| Model           | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-----------------|--------------|----------------|------------|---------------|
| Inception v3    | 91.20        | 91.00          | 90.00      | 90.50         |
| **Inception v4** | **95.14**     | **95.00**        | **95.00**    | **95.20**       |

---

---
