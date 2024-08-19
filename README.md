# Alzheimer-project
## 1. Introdução

  Conhecido comumente como a doença da perda de memória, o Alzheimer é um transtorno neurodegenerativo progressivo que faz com que o paciente, ao longo do tempo, perca gradativamente a memória e que em casos mais avançados tenha emoções, personalidade e comportamentos sociais alterados. Apesar de comumente as pessoas terem conhecimento dessa doença após o agravamento dos sintomas mais simples, é possível observar a probabilidade do indivíduo adquirir a doença ou não baseados em características fisiologicas e rotineiras dela, para isso, foi utilziados dados públicos que possuem essas característica dos pacientes e se a pessoa possuia ou não a doença. 
Os dados foram adquiridos na base de dados público Kaggle.com, segundo o autor os dado não possuem nenhum tipo de preprocesamento de dados e somente as respostas dos pacientes, além disso, para melhor compreensão da criação do modelo, foi utilizado a metodologia CRISP-DM que consiste 6 principais etapas para a criação de um projeto de data science.


## 2. Ferramentas utilizadas
  - Tratamento dos dados: Pandas
  - Biblioteca de Machine Learning: Sklearn
  - Ferramentas de seleção de Features: Chi² e F_classifier 
  - Ferramentas de pré processamentos: SMOTE e Normalize
  - Gráficos: Seaborn e Matplotlib
  - Modelos de ML:
 
## 3. Principais Features e Target
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
 
## 4. Problematização e objetivo do projeto
4.1 Objetivo

  O objetivo deste projeto foi a criação de um modelo preditivo capaz de detectar a possível presença dessa doença utilizando dados como características demográficas, estilo de vida, histórico médico, avaliação funcional e cognitiva, além dos sintomas do paciente. Isso permite que, mesmo na ausência dos sintomas mais leves, o modelo possa identificar se o paciente tende a desenvolver a doença antes do aparecimento dos sintomas mais avançados ou antes de uma cosulta médica.
  O intuito desse projeto é tentar trazer o projeto mais próximo da realidade contento tratamento, preprocessamento, criação do modelo, tuning e deploy do projeto

4.2 Beneficios

  O principal benefício está relacionado com a possível antecipação do indivíduo identificar se há a probabilidade de possuir alzheimer ou não (obs: Vale ressaltar que o cunho desse projeto é para estudos e caso tenha colocado os seus dados para gerar um resultado jamais deverá ser levado como verdade única, caso tenha dúvidas procure um médico especializado da área), outro ponto que pode ser utiziado é trazer mais uma camada de de interpretação para o diagnóstico do paciente pelo médico em relação da presença da doença.


## 5. Estruturação
  5.1 Estrutura do Crisp-DM

  <div align="center">
  <img src="https://github.com/user-attachments/assets/f5b4f818-83b3-42ab-8193-7c994eff5545" width="500px" />
  </div>

  
  5.2 Estruturação do nosso projeto

  Com base na metodologia Crisp-DM, a estruturação do projeto será feita da seguinte maneira:

* Business Understanding
* Data Understanding
  * Separção de dataset treino e teste
  * Exploração de dados
  * Observando as distribuição das fetures
  * Outliers
  * Matriz de Correlação

* Data Preparation

* Modeling
  * Preprocessamento dos dados
    * Busca do Modelo
    * Cross Validation
  * Pipeline
  * Tuning
  * Pipeline + Tuning

* Evaluation
  * Cross Validation
  * Matriz de Confusão
  * Curva-Roc

* Deploy



## 6. Resultados
<div align="center">
<img src="https://github.com/user-attachments/assets/4cc2376e-8639-4dab-b83f-efd595c692e0" width="200px" />
</div>
* Pipeline final contendo os preprocessamentos e o modelo a ser utilizado.



<div align="center">
<img src="https://github.com/user-attachments/assets/20868082-8381-4281-a043-6b6767cf7621" width="150px" />
</div>

* O modelo final após a criação da pipeline chegou em bons resultados com a acurácia média de 89% mais a precisão de e recall do paciente possuir a doença com 86% e 82$ respectivamente, além disso esse valores são exprissios já que o nosso dataset possui um desbalanceamento considerável entre os pacientes, onde a quantidade de pessoas que não possui a doença é superior aos que possuissem.

<div align="center">
<img src="https://github.com/user-attachments/assets/af3eb3b9-b894-432a-b68c-8ffe977cc531" width="500px" />
</div>

* A matriz de confusão consegue mostrar visualmente que o nosso modelo tem a capacidade de distinguir bem os novos dados, além disso, junto com os dados adquiridos no classifier report vemos que o percentual de precisão e recall são validados se baseado na matriz.

<div align="center">
<img src="https://github.com/user-attachments/assets/97b26f72-abac-492f-964d-d4aeaa275b31" width="500px" />
</div>

* A pontuação AUC mostra que o nosso modelo tem um nível execelente para a distinção binários entre os pacientes que possuem a doença do Alzhiemer com quem não tem.

