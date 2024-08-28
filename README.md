# Alzheimer-project
## 1. Introduction

  Commonly known as the memory loss disease, Alzheimer's is a progressive neurodegenerative disorder that gradually causes patients to lose their memory over time. In more advanced stages, it can also alter emotions, personality, and social behavior. Although people typically become aware of this disease after the worsening of milder symptoms, it is possible to observe the probability of an individual developing the disease based on their physiological and routine characteristics. To achieve this, public data containing these characteristics of patients and whether or not they had the disease were used. The data was sourced from the public database Kaggle.com, and according to the author, the data has not undergone any preprocessing and contains only the patients' responses. Additionally, for a better understanding of the model creation process, the CRISP-DM methodology, which consists of six main stages for developing a data science project, was employed.


## 2. Tools
  - Data Processing: Pandas e Numpy
  - library Machine Learning: Scikit-Learn
  - Pre processing tools: ADASYN e StandardScaler
  - Graphics: Seaborn e Matplotlib
  - ML Model: XGBoost
 
## 3. Main Features and Target
Patient's infomation
Patient ID

    PatientID: A unique identifier assigned to each patient (4751 to 6900).

Demographic Details

    Age: The age of the patients ranges from 60 to 90 years.
    Gender: Gender of the patients, where 0 represents Male and 1 represents Female.
    Ethnicity: The ethnicity of the patients, coded as follows:
    0: Caucasian
    1: African American
    2: Asian
    3: Other
    EducationLevel: The education level of the patients, coded as follows:
    0: None
    1: High School
    2: Bachelor's
    3: Higher

Lifestyle Factors

    BMI: Body Mass Index of the patients, ranging from 15 to 40.
    Smoking: Smoking status, where 0 indicates No and 1 indicates Yes.
    AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20.
    PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10.
    DietQuality: Diet quality score, ranging from 0 to 10.
    SleepQuality: Sleep quality score, ranging from 4 to 10.

Medical History

    FamilyHistoryAlzheimers: Family history of Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.
    CardiovascularDisease: Presence of cardiovascular disease, where 0 indicates No and 1 indicates Yes.
    Diabetes: Presence of diabetes, where 0 indicates No and 1 indicates Yes.
    Depression: Presence of depression, where 0 indicates No and 1 indicates Yes.
    HeadInjury: History of head injury, where 0 indicates No and 1 indicates Yes.
    Hypertension: Presence of hypertension, where 0 indicates No and 1 indicates Yes.

Clinical Measurements

    SystolicBP: Systolic blood pressure, ranging from 90 to 180 mmHg.
    DiastolicBP: Diastolic blood pressure, ranging from 60 to 120 mmHg.
    CholesterolTotal: Total cholesterol levels, ranging from 150 to 300 mg/dL.
    CholesterolLDL: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL.
    CholesterolHDL: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL.
    CholesterolTriglycerides: Triglycerides levels, ranging from 50 to 400 mg/dL.

Cognitive and Functional Assessments

    MMSE: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.
    FunctionalAssessment: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.
    MemoryComplaints: Presence of memory complaints, where 0 indicates No and 1 indicates Yes.
    BehavioralProblems: Presence of behavioral problems, where 0 indicates No and 1 indicates Yes.
    ADL: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.

Symptoms

    Confusion: Presence of confusion, where 0 indicates No and 1 indicates Yes.
    Disorientation: Presence of disorientation, where 0 indicates No and 1 indicates Yes.
    PersonalityChanges: Presence of personality changes, where 0 indicates No and 1 indicates Yes.
    DifficultyCompletingTasks: Presence of difficulty completing tasks, where 0 indicates No and 1 indicates Yes.
    Forgetfulness: Presence of forgetfulness, where 0 indicates No and 1 indicates Yes.

Diagnosis Information

    Diagnosis: Diagnosis status for Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.

Confidential Information

    DoctorInCharge: This column contains confidential information about the doctor in charge, with "XXXConfid" as the value for all patients.

 
## 4. Problem Statement and Project's Goals
4.1 Project's Goal

  The goal of this project was to create a predictive model capable of detecting the possible presence of Alzheimer's disease using data such as demographic characteristics, lifestyle factors, medical history, functional and cognitive assessments, as well as patient symptoms. This allows the model to identify whether a patient is likely to develop the disease even in the absence of milder symptoms, before the onset of more advanced symptoms or a medical consultation.

  The purpose of this project is to bring the approach closer to real-world applications by including data treatment, preprocessing, model creation, tuning, and deployment.

4.2 Benefit

  The main benefit is the potential for individuals to identify the likelihood of having Alzheimer's disease in advance (Note: It is important to emphasize that this project is intended for study purposes only, and any results generated from personal data should never be taken as the sole truth. If you have any concerns, please consult a specialized medical professional). Additionally, this model can provide an extra layer of interpretation for a doctor’s diagnosis regarding the presence of the disease

## 5. Structure
  5.1 Structure of Crisp-DM

  <div align="center">
  <img src="https://github.com/user-attachments/assets/f5b4f818-83b3-42ab-8193-7c994eff5545" width="500px" />
  </div>

  
  5.2 Structure of our project

  Using the Crisp-DMs models, the project will be structure by following the steps:

* Business Understanding
* Data Understanding
  * Splitting the dataset of train and test
  * Data Exploration
  * Observing the distribution of the features
  * Outliers
  * Correlation Matrix

* Data Preparation

* Modeling
  * Preprocess of data
    * Search of model
    * Cross Validation
  * Pipeline
  * Tuning
  * Pipeline + Tuning

* Evaluation
  * Nested Cross Validation
  * Confusion Matrix
  * Curve Roc

* Deploy



## 6. Results
<div align="center">
<img src="https://github.com/user-attachments/assets/5c0b9d63-ecee-46d8-bdb9-abfafae01db3" width="200px" />
</div>
* Final Pipeline content the preprocessments and the main model.



<div align="center">
<img src="https://github.com/user-attachments/assets/c0ac6f74-9e2a-453e-b3c5-45ae9ad0ddd2" width="150px" />
</div>

* The final model, after creating the pipeline, achieved good results with an average accuracy of 94%, and precision and recall for patients with the disease of 93% and 92%, respectively. Moreover, these values are accurate given that our dataset has a significant imbalance between patients, with a higher number of individuals without the disease compared to those with it.

<div align="center">
<img src="https://github.com/user-attachments/assets/af3eb3b9-b894-432a-b68c-8ffe977cc531" width="500px" />
</div>

* The confusion matrix visually demonstrates that our model has the ability to distinguish new data effectively. Additionally, together with the data obtained from the classifier report, we can see that the precision and recall percentages are validated based on the matrix.

<div align="center">
<img src="https://github.com/user-attachments/assets/97b26f72-abac-492f-964d-d4aeaa275b31" width="500px" />
</div>

* The AUC score shows that our model has an excellent level of performance in distinguishing between patients who have Alzheimer's disease and those who do not.
