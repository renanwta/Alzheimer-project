# Alzheimer-project
## 1. Introdução

  Commonly known as the memory loss disease, Alzheimer's is a progressive neurodegenerative disorder that gradually causes patients to lose their memory over time. In more advanced stages, it can also alter emotions, personality, and social behavior. Although people typically become aware of this disease after the worsening of milder symptoms, it is possible to observe the probability of an individual developing the disease based on their physiological and routine characteristics. To achieve this, public data containing these characteristics of patients and whether or not they had the disease were used. The data was sourced from the public database Kaggle.com, and according to the author, the data has not undergone any preprocessing and contains only the patients' responses. Additionally, for a better understanding of the model creation process, the CRISP-DM methodology, which consists of six main stages for developing a data science project, was employed.


## 2. Ferramentas utilizadas
  - Data Processing: Pandas e Numpy
  - library Machine Learning: Scikit-Learn
  - Pre processing tools: ADASYN e StandardScaler
  - Graphics: Seaborn e Matplotlib
  - ML Model: XGBoost
 
## 3. Main Features and Target
Informações do Paciente
- ID do Paciente
    - PatientID: Um identificador único atribuído a cada paciente (4751 a 6900).

- Detalhes Demográficos

    - Idade: A idade dos pacientes varia de 60 a 90 anos.
    - Gênero: Gênero dos pacientes, onde 0 representa Masculino e 1 representa Feminino.
    - Etnia: A etnia dos pacientes, codificada da seguinte forma:
        0: Caucasiano
        1: Afro-Americano
        2: Asiático
        3: Outro
    - Nível de Escolaridade: O nível de escolaridade dos pacientes, codificado da seguinte forma:
        0: Nenhum
        1: Ensino Médio
        2: Graduação
        3: Pós-Graduação

- Fatores de Estilo de Vida

    - IMC: Índice de Massa Corporal dos pacientes, variando de 15 a 40.
    - Tabagismo: Status de tabagismo, onde 0 indica Não e 1 indica Sim.
    - Consumo de Álcool: Consumo semanal de álcool em unidades, variando de 0 a 20.
    - Atividade Física: Atividade física semanal em horas, variando de 0 a 10.
    - Qualidade da Dieta: Pontuação da qualidade da dieta, variando de 0 a 10.
    - Qualidade do Sono: Pontuação da qualidade do sono, variando de 4 a 10.

- Histórico Médico

    - Histórico Familiar de Alzheimer: Histórico familiar de Doença de Alzheimer, onde 0 indica Não e 1 indica Sim.
    - Doença Cardiovascular: Presença de doença cardiovascular, onde 0 indica Não e 1 indica Sim.
    - Diabetes: Presença de diabetes, onde 0 indica Não e 1 indica Sim.
    - Depressão: Presença de depressão, onde 0 indica Não e 1 indica Sim.
    - Lesão na Cabeça: Histórico de lesão na cabeça, onde 0 indica Não e 1 indica Sim.
    - Hipertensão: Presença de hipertensão, onde 0 indica Não e 1 indica Sim.

- Medições Clínicas

    - Pressão Arterial Sistólica: Pressão arterial sistólica, variando de 90 a 180 mmHg.
    - Pressão Arterial Diastólica: Pressão arterial diastólica, variando de 60 a 120 mmHg.
    - Colesterol Total: Níveis de colesterol total, variando de 150 a 300 mg/dL.
    - Colesterol LDL: Níveis de colesterol de lipoproteína de baixa densidade, variando de 50 a 200 mg/dL.
    - Colesterol HDL: Níveis de colesterol de lipoproteína de alta densidade, variando de 20 a 100 mg/dL.
    - Triglicerídeos: Níveis de triglicerídeos, variando de 50 a 400 mg/dL.

- Avaliações Cognitivas e Funcionais

    - MMSE: Pontuação do Mini Exame do Estado Mental, variando de 0 a 30. Pontuações mais baixas indicam comprometimento cognitivo.
    - Avaliação Funcional: Pontuação de avaliação funcional, variando de 0 a 10. Pontuações mais baixas indicam maior comprometimento.
    - Queixas de Memória: Presença de queixas de memória, onde 0 indica Não e 1 indica Sim.
    - Problemas Comportamentais: Presença de problemas comportamentais, onde 0 indica Não e 1 indica Sim.
    - Atividades da Vida Diária: Pontuação das atividades da vida diária, variando de 0 a 10. Pontuações mais baixas indicam maior comprometimento.

- Sintomas

    - Confusão: Presença de confusão, onde 0 indica Não e 1 indica Sim.
    - Desorientação: Presença de desorientação, onde 0 indica Não e 1 indica Sim.
    - Mudanças de Personalidade: Presença de mudanças de personalidade, onde 0 indica Não e 1 indica Sim.
    - Dificuldade em Completar Tarefas: Presença de dificuldade em completar tarefas, onde 0 indica Não e 1 indica Sim.
    - Esquecimento: Presença de esquecimento, onde 0 indica Não e 1 indica Sim.

- Informações de Diagnóstico

    - Diagnóstico: Status de diagnóstico para Doença de Alzheimer, onde 0 indica Não e 1 indica Sim.

- Informações Confidenciais

    - Médico Responsável: Esta coluna contém informações confidenciais sobre o médico responsável, com "XXXConfid" como o valor para todos os pacientes.
 
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
  * Preprocessamento dos dados
    * Busca do Modelo
    * Cross Validation
  * Pipeline
  * Tuning
  * Pipeline + Tuning

* Evaluation
  * Nested Cross Validation
  * Matriz de Confusão
  * Curva-Roc

* Deploy



## 6. Resultados
<div align="center">
<img src="https://github.com/user-attachments/assets/5c0b9d63-ecee-46d8-bdb9-abfafae01db3" width="200px" />
</div>
* Pipeline final contendo os preprocessamentos e o modelo a ser utilizado.



<div align="center">
<img src="https://github.com/user-attachments/assets/c0ac6f74-9e2a-453e-b3c5-45ae9ad0ddd2" width="150px" />
</div>

* O modelo final após a criação da pipeline chegou em bons resultados com a acurácia média de 94% mais a precisão de e recall do paciente possuir a doença com 93% e 92% respectivamente, além disso esse valores são precisos já que o nosso dataset possui um desbalanceamento considerável entre os pacientes, onde a quantidade de pessoas que não possui a doença é superior aos que possuissem.

<div align="center">
<img src="https://github.com/user-attachments/assets/af3eb3b9-b894-432a-b68c-8ffe977cc531" width="500px" />
</div>

* A matriz de confusão consegue mostrar visualmente que o nosso modelo tem a capacidade de distinguir bem os novos dados, além disso, junto com os dados adquiridos no classifier report vemos que o percentual de precisão e recall são validados se baseado na matriz.

<div align="center">
<img src="https://github.com/user-attachments/assets/97b26f72-abac-492f-964d-d4aeaa275b31" width="500px" />
</div>

* A pontuação AUC mostra que o nosso modelo tem um nível execelente para a distinção binários entre os pacientes que possuem a doença do Alzhiemer com quem não tem.

