# Alzheimer-project
## Métodologia CRISP-DM
O CRISP-DM é uma metodologia que consiste 6 principais etapas para a criação de um projeto de business, o presente projeto teve como sua base de criação ao modelo CRISP contendo 6 principais etapas:
<div align="center">
<img src="https://github.com/user-attachments/assets/0c9fb82d-15a3-43ce-9a8c-6dafce20b542" width="500px" />
</div>



## Business Understanding
O dados foram adquiridos na base de dados público Kaggle.com, segundo o autor os dado não possuem nenhum tipo de preprocesamento de dados e somente as respostas dos pacientes.

- Objetivo
  O objetivo deste projeto foi a criação de um modelo preditivo capaz de detectar a possível presença dessa doença utilizando dados como características demográficas, estilo de vida, histórico médico, avaliação funcional e cognitiva, além dos sintomas do paciente. Isso permite que, mesmo na ausência dos sintomas mais leves, o modelo possa identificar se o paciente tende a desenvolver a doença antes do aparecimento dos sintomas mais avançados ou antes de uma cosulta médica.

- Ferramentas
  - Tratamento dos dados: Pandas
  - Biblioteca de Machine Learning: Sklearn
  - Ferramentas de seleção de Features: Chi² e F_classifier 
  - Ferramentas de pré processamentos: SMOTE e Normalize
  - Gráficos: Seaborn e Matplotlib
  - Modelos de ML:
 
## Data Understanding
- Patient Information
  - Patient ID

    PatientID: A unique identifier assigned to each patient (4751 to 6900).

- Demographic Details

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

- Lifestyle Factors

    BMI: Body Mass Index of the patients, ranging from 15 to 40.
    Smoking: Smoking status, where 0 indicates No and 1 indicates Yes.
    AlcoholConsumption: Weekly alcohol consumption in units, ranging from 0 to 20.
    PhysicalActivity: Weekly physical activity in hours, ranging from 0 to 10.
    DietQuality: Diet quality score, ranging from 0 to 10.
    SleepQuality: Sleep quality score, ranging from 4 to 10.

- Medical History

    FamilyHistoryAlzheimers: Family history of Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.
    CardiovascularDisease: Presence of cardiovascular disease, where 0 indicates No and 1 indicates Yes.
    Diabetes: Presence of diabetes, where 0 indicates No and 1 indicates Yes.
    Depression: Presence of depression, where 0 indicates No and 1 indicates Yes.
    HeadInjury: History of head injury, where 0 indicates No and 1 indicates Yes.
    Hypertension: Presence of hypertension, where 0 indicates No and 1 indicates Yes.

- Clinical Measurements

    SystolicBP: Systolic blood pressure, ranging from 90 to 180 mmHg.
    DiastolicBP: Diastolic blood pressure, ranging from 60 to 120 mmHg.
    CholesterolTotal: Total cholesterol levels, ranging from 150 to 300 mg/dL.
    CholesterolLDL: Low-density lipoprotein cholesterol levels, ranging from 50 to 200 mg/dL.
    CholesterolHDL: High-density lipoprotein cholesterol levels, ranging from 20 to 100 mg/dL.
    CholesterolTriglycerides: Triglycerides levels, ranging from 50 to 400 mg/dL.

- Cognitive and Functional Assessments

    MMSE: Mini-Mental State Examination score, ranging from 0 to 30. Lower scores indicate cognitive impairment.
    FunctionalAssessment: Functional assessment score, ranging from 0 to 10. Lower scores indicate greater impairment.
    MemoryComplaints: Presence of memory complaints, where 0 indicates No and 1 indicates Yes.
    BehavioralProblems: Presence of behavioral problems, where 0 indicates No and 1 indicates Yes.
    ADL: Activities of Daily Living score, ranging from 0 to 10. Lower scores indicate greater impairment.

- Symptoms

    Confusion: Presence of confusion, where 0 indicates No and 1 indicates Yes.
    Disorientation: Presence of disorientation, where 0 indicates No and 1 indicates Yes.
    PersonalityChanges: Presence of personality changes, where 0 indicates No and 1 indicates Yes.
    DifficultyCompletingTasks: Presence of difficulty completing tasks, where 0 indicates No and 1 indicates Yes.
    Forgetfulness: Presence of forgetfulness, where 0 indicates No and 1 indicates Yes.

- Diagnosis Information

    Diagnosis: Diagnosis status for Alzheimer's Disease, where 0 indicates No and 1 indicates Yes.

- Confidential Information

    DoctorInCharge: This column contains confidential information about the doctor in charge, with "XXXConfid" as the value for all patients.
