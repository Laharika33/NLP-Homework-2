# README: Bigram Language Model & Confusion Matrix Metrics

**Student Name:** Laharika Dattu  
**Student ID:** 700788001  
**Course:** Natural Language Processing

## Overview
This repository contains two Python programs that demonstrate fundamental concepts in **Natural Language Processing (NLP)** and **Machine Learning evaluation metrics**:

1. **Bigram Language Model** – Builds a probabilistic model of text using bigram counts and computes sentence probabilities.
2. **Confusion Matrix Metrics** – Computes precision and recall metrics for multi-class classification, including per-class, macro-averaged, and micro-averaged scores.

These programs are designed for educational purposes and illustrate key concepts in NLP and classification evaluation.

---

## Program 1: Bigram Language Model

### File: `bigram_model.py`

### Description
Constructs a **bigram language model** from a small training corpus. It calculates **unigram counts**, **bigram counts**, and **bigram probabilities** using Maximum Likelihood Estimation (MLE). It also evaluates the probability of test sentences and compares which sentence the model prefers.

### Features
- Tokenizes sentences into words.
- Computes unigram and bigram counts.
- Computes bigram probabilities using MLE.
- Calculates the probability of given sentences.
- Compares sentence probabilities to determine preference.

### Usage
1. Update the `corpus` list with your training sentences.
2. Run the script:
```bash
python bigram_model.py
```
3. Output includes:
   - Unigram counts  
   - Bigram counts  
   - Bigram probabilities  
   - Probabilities of test sentences  
   - Preferred sentence

### Example Output
```
=== Unigram Counts ===
<s>: 3, I: 2, love: 2, NLP: 1, deep: 2, learning: 2, is: 1, fun: 1, </s>: 3
=== Bigram Probabilities (MLE) ===
P(I|<s>) = 0.667, P(deep|<s>) = 0.333
...
Sentence 1: <s> I love NLP </s> → Probability: 0.167
Sentence 2: <s> I love deep learning </s> → Probability: 0.083
 The model prefers Sentence 1
```

### Dependencies
- Python 3.x  
- No external libraries required

---

## Program 2: Confusion Matrix Metrics

### File: `confusion_matrix_metrics.py`

### Description
Calculates **precision** and **recall** metrics from a multi-class **confusion matrix**. Computes per-class metrics and both **macro-averaged** and **micro-averaged** metrics.

### Features
- Computes **per-class precision and recall**.
- Computes **macro-averaged metrics** (equal weight to all classes).
- Computes **micro-averaged metrics** (aggregates counts across classes).
- Handles any number of classes.

### Usage
1. Define your confusion matrix using NumPy (rows = predicted labels, columns = actual labels).  
2. Set the `classes` list with your class names.  
3. Run the script:
```bash
python confusion_matrix_metrics.py
```
4. Output includes:
   - Per-class precision and recall  
   - Macro-averaged precision and recall  
   - Micro-averaged precision and recall

### Example Output
```
Cat -> Precision: 0.250, Recall: 0.250
Dog -> Precision: 0.500, Recall: 0.444
Rabbit -> Precision: 0.333, Recall: 0.400
Macro-averaged Precision: 0.361
Macro-averaged Recall: 0.365
Micro-averaged Precision: 0.386
Micro-averaged Recall: 0.386
```

### Dependencies
- Python 3.x  
- NumPy (`pip install numpy`)

---

## Notes
- **Bigram Language Model:** Unseen bigrams result in probability 0. Smoothing (e.g., Laplace) can be added for robustness.  
- **Confusion Matrix Metrics:** Macro vs. Micro averaging gives different perspectives on classifier performance, useful for imbalanced datasets.


