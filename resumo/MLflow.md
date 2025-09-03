
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
