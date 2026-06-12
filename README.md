# FredCare AI: Breast Cancer Prediction System

**A production ready clinical decision support web application powered by machine learning, built for the early detection of breast cancer malignancy.**

***

## Table of Contents

* Overview
* Why This Project Matters
* The Dataset
* The Model
* Application Features
* How to Run
* Project Structure
* Results
* Disclaimer

***

## Overview

Breast cancer is one of the most commonly diagnosed cancers in the world and one of the leading causes of death among women across all age groups and geographic regions. Early and accurate detection dramatically increases survival rates, reduces the need for aggressive medical intervention, and gives patients the opportunity to make timely and informed decisions about their treatment.

FredCare AI was built to demonstrate how machine learning can be applied in a real world healthcare context to assist medical professionals in distinguishing malignant tumors from benign ones. The project combines a rigorous, well documented data science workflow with a clean, intuitive web interface that any clinician, researcher, or student can use to submit clinical measurements and receive an instant prediction with a confidence score and a professional recommendation.

***

## Why This Project Matters

### The Medical Problem

Cancer detection is a time sensitive discipline. The difference between identifying a tumor as malignant versus benign can determine whether a patient follows a watchful monitoring plan or undergoes urgent surgical intervention. Traditional diagnostic methods such as biopsies, imaging, and laboratory analysis are accurate but slow, costly, and not always immediately accessible in every clinical setting.

Machine learning models trained on well established medical datasets can serve as a powerful first line screening tool. They do not replace specialist diagnosis; they complement it by quickly flagging high risk cases for priority review, reducing the time between a concerning measurement and an expert consultation.

### Why We Built This Model

The creation of this model was driven by five core motivations:

* **Speed:** A trained model returns a prediction in milliseconds, enabling rapid initial screening without waiting for laboratory or imaging results.

* **Accessibility:** Medical facilities with limited access to specialist imaging equipment can use a measurement based tool as a preliminary filter to prioritise urgent cases.

* **Consistency:** Unlike a subjective visual assessment, a statistical model applies exactly the same decision criteria to every patient, removing variability introduced by fatigue, inexperience, or cognitive bias.

* **Education:** This project serves as a complete, end to end example of how to design, train, evaluate, and deploy a healthcare machine learning solution in a responsible and transparent manner.

* **Interpretability:** Logistic Regression is one of the most transparent algorithms in applied machine learning. Every prediction can be traced back to the individual weight each clinical feature contributes to the final decision. This transparency is essential in healthcare, where clinicians must be able to scrutinise and justify automated recommendations.

***

## The Dataset

This project uses the **Wisconsin Breast Cancer Dataset**, one of the most widely studied and well validated benchmarks in the machine learning community. It was originally curated by Dr. William H. Wolberg at the University of Wisconsin Hospital and is available directly through the `sklearn.datasets` module in Python.

Each record in the dataset represents a digitised image of a fine needle aspirate (FNA) of a breast mass. The numerical features describe measurable characteristics of the cell nuclei present in the image, including their physical size, geometric shape, surface texture, and structural irregularity.

### Dataset Summary

| Property | Value |
|---|---|
| Total samples | 569 |
| Malignant cases | 212 (37.3%) |
| Benign cases | 357 (62.7%) |
| Total available features | 30 |
| Features selected for this model | 10 |
| Target classes | Malignant (0), Benign (1) |

### The 10 Selected Features

The ten features used in this model were selected based on the magnitude of their logistic regression coefficients, meaning they are the measurements that contribute the most to the model's predictions:

| Group | Feature | Clinical Meaning |
|---|---|---|
| Size | Mean Radius | Average distance from the nucleus centre to its boundary |
| Size | Mean Perimeter | Mean perimeter of the cell nucleus |
| Size | Mean Area | Mean area enclosed within the nucleus boundary |
| Size | Mean Texture | Standard deviation of grey scale values in the nucleus |
| Shape | Mean Smoothness | Local variation in nucleus radius lengths |
| Shape | Mean Compactness | Perimeter squared divided by area, minus 1.0 |
| Shape | Mean Symmetry | Symmetry of the nucleus shape |
| Shape | Mean Fractal Dimension | Coastline approximation of the nucleus boundary |
| Concavity | Mean Concavity | Severity of concave portions of the nucleus contour |
| Concavity | Mean Concave Points | Number of concave portions of the nucleus contour |

***

## The Model

### Algorithm Choice

**Logistic Regression** was selected as the core classification algorithm. This choice was deliberate and grounded in the specific demands of clinical applications:

* It is highly interpretable. Each feature carries an explicit numerical coefficient that directly explains its positive or negative contribution to the malignancy prediction.
* It produces well calibrated class probabilities, which are essential in clinical settings where the confidence of a prediction matters as much as the label itself.
* It performs with excellent accuracy on linearly separable medical datasets such as this one.
* It is computationally lightweight and suitable for real time, browser based deployment without requiring specialist hardware.

### Training Architecture

The pipeline uses two separate models trained from the same data:

**1. Evaluation Model**

Trained on 80% of the dataset using a stratified split (random state 42) and tested on the remaining 20% (114 samples). This model is used exclusively to compute honest, held out performance metrics. Stratification ensures the proportion of malignant and benign cases is preserved in both the training and test portions.

**2. Production Model**

Trained on the full dataset of 569 samples. This model is the one that powers all live predictions in the FredCare AI web application. Training on the full dataset ensures that every available clinical observation contributes to the inference process.

### Preprocessing

All input features are standardised using `StandardScaler` before being passed to the model. The scaler is fitted exclusively on the training portion to prevent data leakage. In the production setting, the scaler is fitted on the complete dataset before being saved and applied to new patient inputs at inference time.

### Cross Validation

A 5 fold stratified cross validation is applied to the evaluation model to confirm that the reported accuracy is not the result of a favourable random split. Each fold preserves the original class distribution, giving a robust and reliable estimate of real world performance.

***

## Application Features

FredCare AI is built with **Streamlit** and delivers a professional, responsive clinical web interface with the following structure:

### Sidebar: Patient Input Panel

The sidebar contains 10 numerical input fields, organised into three clinically meaningful groups. All fields are pre filled with the dataset median value so that clinicians have an immediate reference point for a typical measurement.

**📐 Size Measurements**
* Mean Radius
* Mean Perimeter
* Mean Area
* Mean Texture

**🔷 Shape and Texture**
* Mean Smoothness
* Mean Compactness
* Mean Symmetry
* Mean Fractal Dimension

**🌀 Concavity**
* Mean Concavity
* Mean Concave Points

An automatic range validation alerts the clinician if any entered value falls significantly outside the expected distribution for that feature.

### Main Panel: Screening Results

After clicking **Run Analysis**, the main panel displays three output sections:

**1. Result Banner**
A full width colored card showing the diagnosis, either MALIGNANT or BENIGN, together with the overall model confidence score expressed as a percentage. The color of the banner changes based on the diagnosis: red for malignant, green for benign.

**2. Screening Details**
Two information cards displayed side by side. The first shows the confidence score as a large number with a visual progress bar. The second shows the full probability breakdown, displaying the individual probability the model assigns to each class.

**3. Clinical Assessment and Recommendation**
A professionally written clinical recommendation tailored to both the predicted class and the confidence tier (high, moderate, or low). The note advises the appropriate next steps, ranging from immediate oncologist referral for high confidence malignant predictions to routine annual imaging for high confidence benign results.

***

## How to Run

### Prerequisites

Ensure Python 3.8 or above is installed. Then install all required libraries using the following command:

```
pip install streamlit scikit-learn pandas numpy
```

### Launching the Web Application

```
streamlit run app.py
```

The application will open automatically in your default browser at:

```
http://localhost:8501
```

### Running the Analysis Notebook

Open `breast_cancer_prediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run all cells in order. The notebook covers the full analytical workflow:

* Data loading and descriptive statistics
* Class distribution analysis
* Feature correlation and selection
* Model training and coefficient analysis
* Evaluation on the held out test set
* 5 fold cross validation
* Confusion matrix and performance metrics
* Conclusions and clinical interpretation

***

## Project Structure

```
CodeAlpha_Breast_Cancer_Prediction/
│
├── app.py                           Streamlit web application (FredCare AI)
├── breast_cancer_prediction.ipynb   Full machine learning analysis notebook
└── README.md                        Project documentation
```

***

## Results

The model achieves the following performance on the held out test set, which consists of 114 samples never seen during training:

| Metric | Score |
|---|---|
| Accuracy | 97.37% |
| ROC AUC Score | 0.9954 |
| Cross Validation Mean Accuracy | 97.80% |
| Cross Validation Score Range | 95.65% to 100.00% |
| False Negatives on Test Set | 3 |

These results confirm that the model generalises well beyond the training data and maintains high precision and recall across both the malignant and benign classes.

A ROC AUC score of 0.9954 indicates that the model is able to distinguish between the two classes with near perfect reliability across all possible decision thresholds, which is a strong result for a clinical screening tool.

***

## Disclaimer

FredCare AI is an educational and research project developed as part of an academic machine learning curriculum. It is **not** a certified or approved medical device.

This application must **not** be used as a substitute for professional medical diagnosis, clinical judgement, radiological review, pathological confirmation, or specialist consultation of any kind.

Any prediction produced by this tool is probabilistic in nature and carries a margin of error. It should be interpreted exclusively as a supplementary signal within a broader clinical evaluation conducted by a qualified healthcare professional.

The author accepts no liability for any clinical decision made on the basis of output from this application.

***

*Developed by Adossi Fred William, African Leadership University*

*Submitted as part of the CodeAlpha Machine Learning Internship Program*
