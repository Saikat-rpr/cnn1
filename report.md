# Assignment: Fork-Based Collaborative CNN Development

## 1. Introduction

This mini project was designed to practise both convolutional neural network (CNN) training and GitHub collaboration. Two users worked on related image classification tasks using different datasets and separate models. Instead of sharing raw data, only trained models and evaluation metrics were exchanged. This report summarises the datasets, models, training setup and cross-dataset evaluation results.

---

## 2. Datasets

### 2.1 User 1 – Animals Dataset - https://www.kaggle.com/datasets/antobenedetti/animals

User 1 used an Animals dataset with five classes:

- cat  
- dog  
- elephant  
- horse  
- lion  

The dataset was organised in the following structure:

```text
data/user1_animals/
  train/<5 classes>/
  val/<5 classes>/
  test/<5 classes>/
```

The test set has only a few images per class, so single errors affect accuracy strongly.

### 2.2 User 2 – PlantVillage Subset - https://www.kaggle.com/datasets/emmarex/plantdisease

User 2 used a 5-class subset of the PlantVillage dataset:

- Pepper__bell___Bacterial_spot  
- Potato___Early_blight  
- Tomato__Target_Spot  
- Tomato__Tomato_mosaic_virus  
- Tomato_Leaf_Mold  

The dataset was split using a 70/15/15 ratio:

```text
data/user2_plants/
  train/<5 classes>/
  val/<5 classes>/
  test/<5 classes>/
```

---

## 3. Models

### 3.1 Model v1 – SimpleCNN for Animals (User 1)

Implemented in `models/model_v1.py`:

- 3 convolutional blocks  
  - Conv2d + BatchNorm + ReLU  
  - MaxPool after each block  
- Adaptive average pooling  
- Classifier:
  - Linear(128 → 256), ReLU, Dropout(0.5)
  - Linear(256 → 5)

Trained only on the animal dataset.

### 3.2 Model v2 – SimpleCNNv2 for PlantVillage (User 2)

Implemented in `models/model_v2.py`:

- 5 ConvBlocks (32 → 64 → 96 → 128 channels)  
- BatchNorm + LeakyReLU  
- Dropout2d inside blocks  
- Global average pooling  
- Classifier:
  - Linear(128 → 128), LeakyReLU, Dropout(0.4)
  - Linear(128 → 5)

Trained only on the PlantVillage subset.

---

## 4. Training Setup

### 4.1 Common Settings

- Input: 256×256 resize → 224×224 crop  
- Normalisation: ImageNet mean/std  
- Loss: CrossEntropy  
- Optimiser: Adam  
- Epochs: 10  
- Device: GPU if available  

### 4.2 User 1 Training (Animals)

Augmentation:

- RandomResizedCrop(224)  
- RandomHorizontalFlip  

Model saved as `models/model_v1.pth`.  
Metrics saved in `results/metrics_v1.json`.

### 4.3 User 2 Training (PlantVillage)

Augmentation:

- Horizontal & vertical flips  
- RandomRotation(15°)  
- RandomResizedCrop(224, scale=(0.8, 1.0))  
- ColorJitter  

Scheduler: StepLR (step=5, gamma=0.5)

Model saved as `models/model_v2.pth`.  
Metrics saved in `results/metrics_v2.json`.

---

## 5. Results

### 5.1 Performance on Own Datasets

| Model      | Dataset               | Test Accuracy |
|------------|------------------------|----------------|
| model_v1   | Animals (User 1)       | ≈ 0.40        |
| model_v2   | PlantVillage (User 2)  | ≈ 0.97        |

### 5.2 Cross-Dataset Evaluation

Cross-testing:

- User 2 ran: `test_model_v1_on_user2.py`
- User 1 ran: `test_model_v2_on_user1.py`

| Trained Model | Train Dataset   | Test Dataset   | Accuracy |
|---------------|-----------------|----------------|----------|
| model_v1      | Animals         | Animals        | ≈ 0.40   |
| model_v1      | Animals         | PlantVillage   | ≈ 0.13   |
| model_v2      | PlantVillage    | PlantVillage   | ≈ 0.97   |
| model_v2      | PlantVillage    | Animals        | ≈ 0.20   |

Domain shift causes large performance drops.

We additionally generated Grad-CAM visualizations to inspect what regions the models focus on (see results/gradcam_* folders in the repository).

---

## 6. Discussion

Both models perform well on their own domains (especially model v2).  
However, cross-dataset accuracy drops heavily due to:

- different colours  
- different textures  
- different backgrounds  
- different shapes  

Model v2 (deeper) generalises slightly better to animals (~0.20 vs ~0.13), but still poorly.  
Small test size in the animal dataset also makes results sensitive to single mistakes.

---

## 7. Git & Collaboration Workflow

- User 1 created the main repository with structure, `.gitignore`, `README.md`, etc.  
- User 1 worked on branch `dev_user1` and made a pull request.  
- User 2 forked the repo, created branch `dev_user2`, added:
  - `model_v2.py`
  - `prepare_user2_plants.py`
  - `train_v2.py`
  - `metrics_v2.json`
- Only model weights were shared, not raw datasets.  
- Cross-testing scripts documented how each model performed on the other dataset.  
- A GitHub issue summarised the combined results.

---

## 8. Conclusion

This project combined CNN-based classification with collaborative GitHub workflows.  
Model v1 and model v2 were trained on animal and plant datasets respectively, then cross-evaluated. Results show:

- average-to-strong performance on own datasets  
- very limited transfer to other domains  

The project also practised using branches, pull requests and clear repository organisation.

