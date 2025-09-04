
## Experimentos
Um experimento pode ter um $n$ número de execuções, onde uma excução é um unica execução de um pedaço de código ou um módelo de aprendizado de máquina.

**Cada execução pode registrar versão do código, hiperparametros, trilhas de matriz, arterfator** isso tudo é armazenado no MLFlow.

Experiementos são agrupamentos lógicos de execuções. Que nos permitem organizar e analisar grupos de execuções.
  
**Sempre quanto cria e exceta um experiemento ele recebe um id=próprio** e um nome que pode ser utilizado para recuperar metadados

### Criando experimentos

```python
# Cria um novo experimento
mlflow.create_experiment() # ela retorna o id do experimento
```
Parametros:
* `name` = defini o nome experimento **deve ser único**
* `artifact_location` = defini o local dos artefartos onde ficará armazenados . **Opicional**
*  `tags` =  mais adiante

```python
# Defini um novo experimento já existente
mlflow.set_experiment()
```
## Logging functions

São funções utilizadas para registrar (logar) informações relevantes durante o treinamento e avaliação de modelos de aprendizado de máquina. Essas informações incluem métricas, parâmetros, artefatos (como gráficos ou arquivos) e modelos, permitindo rastrear experimentos e facilitar reprodutibilidade.

1. As duas primeiras são :

    * Uri de rastramento de conj. de pontos do MLflow e Uri de rastramento de ponto do MLflow

    ```python
mlflow.set_tracking_uri()
    ```

    > É usada para definir o local de sua escolha, onde deseja manter os rastros de seu codigo. A
  
```python
mlflow.get_tracking_uri()
```

    > É recuperar o caminho da localização

exemplo abaixo:
```python
mlflow.set_tracking_uri(
    uri="./works"
)

print("Uri :", mlflow.get_tracking_uri())
try:
    exp = mlflow.create_experiment(
        name = "GradientBoss",
        tags = dict(version_ = "1.0.0")
    )
    get_ = mlflow.get_experiment(experiment_id=exp)

except mlflow.exceptions.MlflowException as e:
    get_ = mlflow.set_experiment(experiment_name="GradientBoss")
```

## Auto logging

É uma ferramenta que permite logging automatico de certos parametros, metricas, artifacts. Sem a necessdade de instrumentação explicita do código.
>  Bom quando o projeto está ficando grande.

* `mlflow.autolog` : Auto logging generico para qualquer lib
* `mlflow.<lib>.autolog` : autologo especifico para algumas libs

Parametros:
1. log_models : log de model ou não
2. log_input_example : se definir True ele os exemplos de entrado para treinamento 

## Tracking Server

É centralização de repositório que armazena metadados e geração de artefatos durante o treinamento de modelos de ml.
* Permite compartilhar resultados com a equipe
Existem dois tipos servidores:
1. Storage : Permite armazenar os artefatos, metadados gerados durante o processo de treinamento
	1. Backend : Usa bando de dados e arquivo de amazenamento
		* SQLlite, Azure, GCP
		 *  Armazena metadados, nome, id, paramentros, metricas , tags, run name e etc
	2. Artefato storage 
		* Input data, output data, modelos ou visuals
		* Local
		
2. Network : todos os usuarios interagem com tracking server usando api rest ou requisição
	1. Rest Api
		* Oferece uma interface simples e flexivel para acessar o servidor de rastreamento por http
	2. RPC
		*  Comunicação bidirecional, mais rapido
	*
Exemplo abaixo usando SQLlite como backend para salvar os dados
	
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mytracks --host 127.0.0.1
```
> `mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")`
> **Remova depois de rodar o código o ` --default-artifact-root ./mytracks`**

## Model Compenent

É um formato padrão para para empacotar seu modelo de aprendizado de máquina em um formato reutilizável. 
> Empacotados em uma formato adequado por exemplo pickle.


* Model Signature: Especifica entrada e saida
	* É uma forma de descrever os tipos de dados de entrada e saida e as formas esperadas e produzidas por um modelo de aprendizado de maquina.
* Model API : Podemos gerar uma api com muita facilidade , e interagir com o model por meio de uma interface
	* REST API : Flask, ou ferramentas python
	* Poder fazer o deploy em diversos sistemas - cloud, on-premises servers, devices
* Flavor : Uma variante refere-se a uma maneira especifica de serializar e armazenar um modelo de aprendizado de máquina.


---

# 📌 `mlflow.evaluate()`

**Definição:**  
A função `mlflow.evaluate()` permite **avaliar um modelo registrado no MLflow** usando um dataset de teste, calculando métricas automaticamente, gerando gráficos de desempenho e salvando artefatos de avaliação no MLflow Tracking.

**Parâmetros principais:**

- `model`: URI do modelo registrado no MLflow (`runs:/<run_id>/<artifact_path>`), objeto PyFunc, ou função custom de predição.
    
- `data`: DataFrame contendo **features + target**.
    
- `targets`: Nome da coluna de target no DataFrame.
    
- `model_type`: Tipo do modelo: `"classifier"` ou `"regressor"`.
    
- `evaluator_config` (opcional): Configurações extras, como lista de labels para classificação multiclass.
    

**Retorno:**  
Um objeto `EvaluationResult` com métricas (`metrics`), artefatos (`artifacts`) e informações sobre o dataset.

---

## 🧪 Exemplo simples com classificação

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Dataset
data = load_wine(as_frame=True)['data']
target = load_wine(as_frame=True)['target']
X = pd.concat([data, target.rename("target")], axis=1)

train_df, test_df = train_test_split(X, test_size=0.25, random_state=42)
X_train, y_train = train_df.drop(columns=["target"]), train_df["target"]

with mlflow.start_run() as run:
    # Treinar e logar modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Obter URI do modelo logado
    model_uri = f"runs:/{run.info.run_id}/model"

    # Avaliar modelo
    eval_result = mlflow.evaluate(
        model=model_uri,
        data=test_df,
        targets="target",
        model_type="classifier",
        evaluator_config={"label_list": [0, 1, 2]}
    )

print(eval_result.metrics)
```

**✅ Saída esperada:**

```python
{'accuracy': 0.955, 'f1_score': 0.954, 'recall': 0.95}
```

---

Se quiser, posso fazer **uma versão em Markdown ainda mais resumida**, tipo “quick reference” para usar `mlflow.evaluate()` em qualquer projeto de classificação ou regressão.

Quer que eu faça?