# Solução de Cinemática Inversa para Robôs 6-DOF via Meta-heurísticas

Este projeto implementa e compara diferentes algoritmos de otimização global (meta-heurísticas) aplicados ao problema de Cinemática Inversa (IK) de um braço robótico industrial de 6 graus de liberdade (6-DOF). O objetivo consiste em determinar os parâmetros articulares ($\theta_1$ a $\theta_6$) capazes de minimizar o erro entre a posição desejada e a posição obtida pelo modelo cinemático direto do robô, respeitando os limites físicos das juntas.

O projeto conta com uma análise estatística de convergência gráfica e uma visualização 3D interativa em tempo real via navegador.

---

## Algoritmos Comparados

Para garantir uma comparação justa, todos os algoritmos partilham o mesmo orçamento computacional baseado em Avaliações de Função Objetivo (FEs):

* **GA (Genetic Algorithm):** Algoritmo Genético baseado em seleção por torneio, cruzamento aritmético e mutação gaussiana.
* **PSO (Particle Swarm Optimization):** Otimização por Enxame de Partículas com peso de inércia e guias cognitivos/sociais.
* **BAS (Beetle Antennae Search):** Algoritmo do Besouro, focado em busca local rápida inspirada no olfato dos besouros.
* **Híbrido (GA-BAS):** Um algoritmo memético que combina a exploração global do GA com a refinação local intensa do BAS.

---

## Estrutura do Projeto

O código foi componentizado em módulos separados para facilitar a manutenção e a escalabilidade:

* **robot_model.py:** Define o modelo DH do robô, cálculo da Cinemática Direta e o mecanismo de reparo de juntas.
* **evaluator.py:** Implementa a classe de rastreamento do custo computacional e guarda a melhor solução global encontrada.
* **algorithms.py:** Contém a implementação de todas as meta-heurísticas (GA, PSO, BAS, GA-BAS).
* **visualizer.py:** Gerencia o servidor do Meshcat, monta a cena 3D com a tríade de eixos e executa o loop de animação.
* **main.py:** Arquivo principal que executa as rodadas estatísticas do experimento, gera os gráficos e inicia a animação.
* **requirements.txt:** Arquivo de definição das dependências do projeto.

---

## Como Funciona o Core Técnico

### 1. Modelo Cinemático
Utiliza matrizes de transformação homogéneas baseadas em parâmetros DH para mapear a base até ao efetuador do manipulador.

### 2. Reparo por Memória Elástica
Em vez de rejeitar soluções inválidas ou simplesmente truncar os limites, o robô aplica uma força elástica que puxa a junta levemente de volta para o centro seguro da sua amplitude física, conforme a equação:

$$\theta_{repaired} = 0.8 \times \theta_{clipped} + 0.2 \times \theta_{mid}$$

### 3. Visualização Assíncrona
A animação 3D roda numa thread separada. Isto permite que possa pausar ou parar o robô pelo terminal sem travar a renderização no navegador.

---

## Como Executar o Projeto

Siga as instruções abaixo para configurar o ambiente e executar os testes de forma isolada e segura.

### 1. Acessar a Pasta do Projeto
Abra o seu terminal na pasta raiz onde os ficheiros do projeto foram guardados.

### 2. Criar um Ambiente Virtual (Venv)
Para evitar conflitos de dependências no seu Python global, configure um ambiente virtual para este projeto.

**No Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```
No Windows (PowerShell):
```powerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
### 3. Instalar as Dependências
Com o ambiente virtual ativado, instale as bibliotecas exigidas executando:

```bash
pip install -r requirements.txt
```
### 4. Rodar a Simulação
Execute o script principal para rodar os testes e abrir o ambiente gráfico:

```bash
python main.py
```
## Controles da Animação
Após a conclusão do benchmark de 20 rodadas, o gráfico de convergência do Matplotlib será exibido e uma aba do seu navegador padrão abrirá automaticamente no servidor do Meshcat (geralmente sob o endereço http://127.0.0.1:7000/static/).

Vá até ao terminal do seu console para controlar o fluxo da animação:

Tecla ENTER (vazio): Alterna o estado da animação entre Pausado e Reproduzindo.

Tecla q + ENTER: Encerra o loop de animação e finaliza o script com segurança.
