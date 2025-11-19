# Collaborative CNN Project

This repository contains the final code and results for a two-user CNN classification project.  
The goal is to demonstrate correct GitHub collaboration using **forks, branches, pull requests, issues** and **cross-evaluation** of models trained on different datasets.

---

## 1. GitHub Workflow (Required Deliverables)

### • Base Repository (User 1)
https://github.com/Saikat-rpr/cnn1

### • User 2 Fork
https://github.com/akhilqt/cnn1

### • Pull Request #1 (User 1 → main)
https://github.com/Saikat-rpr/cnn1/pull/1

### • Pull Request #2 (User 2 → main)
https://github.com/Saikat-rpr/cnn1/pull/2

These links show the full collaborative workflow.

---

## 2. Tasks Completed by Each User

### **User 1**
- Created the main repository  
- Implemented and trained **model_v1** on the Animals dataset  
- Logged results and metrics  
- Evaluated User 2’s model on User 1 dataset  
- Merged PRs and finalized the repository structure  

### **User 2**
- Forked the base repo  
- Worked on branch `dev_user2`  
- Implemented and trained **model_v2** on PlantVillage dataset  
- Logged metrics and opened PR #2  
- Reported cross-testing results through a GitHub Issue  

---

## 3. Repository Structure

```text
cnn1/
  models/
    model_v1.py
    model_v2.py
  notebooks/
    train_v1.py
    train_v2.py
    test_v1_user2.py
    test_v2_user1.py
  results/
    metrics_v1.json
    metrics_v2.json
    test_v1_user2.json
    test_v2_user1.json
  utils/
    metrics.py
    prepare_user2_plants.py
  report.md
  README.md
  requirements.txt
```


---

## 4. Where to Find Important Files

- **Training & testing scripts:** `notebooks/`
- **Models:** `models/`
- **Evaluation metrics:** `results/`
- **Utility scripts:** `utils/`
- **Final written report:** `report.md`

---

## 5. Issue Tracking

User 2 opened an issue to report the performance of User 1’s model on the PlantVillage dataset as part of the workflow requirement.

---

## 6. Summary

This repository completes all required steps of the collaborative CNN assignment, including:
- fork & branch workflow  
- two pull requests  
- issue reporting  
- cross-dataset evaluation  
- final organized repository structure

