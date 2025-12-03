
# Projeto Final — Deep Learning: Aplicação de Redes Neurais Informadas pela Física (PINNs) na Solução da Equação de Wheeler–DeWitt em um Modelo quântico-cosmológico

**Autor:** António Monteiro
**Matrícula:** [DO2520057]
**Curso:** Doutorado em Modelagem Computacional, UERJ/IPRJ (Nova Friburgo)

---

## 🚀 Descrição do Projeto (Spectrum Discovery PINN)

Este projeto implementa **Redes Neurais Informadas pela Física (PINNs)** para resolver o problema de autovalores associado à **equação de Wheeler–DeWitt estacionária** de um modelo cosmológico quântico. [cite_start]O objetivo é comparar os resultados obtidos pela PINN com os valores de referência do Método Espectral de Galerkin (MSG)[cite: 8].

A metodologia central é tratar os **autovalores ($E_n$) como parâmetros treináveis** da rede neural, permitindo que o PINN descubra simultaneamente as autofunções ($\eta_n(a)$) e o espectro de energia, apenas pela imposição das leis da física via função de perda.

---

## 1. Modelo Físico e Equação de Autovalor

O modelo cosmológico quântico, que inclui radiação, constante cosmológica negativa ($\Lambda < 0$) e potencial de Pöschl-Teller, se reduz ao seguinte problema de autovalor para soluções estacionárias:

$$
-\frac{d^2\eta}{da^2} + V(a)\,\eta(a) = 12E\,\eta(a), \quad a\in[0,L],
$$

[cite_start]com condições de contorno de Dirichlet homogêneas ($\eta(0) = \eta(L) = 0$)[cite: 77, 93].

O potencial efetivo $V(a)$ é dado por:
$$
V(a) = 36a^2 + 12|\Lambda|a^4 + 12 a V_0 \,\text{sech}^2(a)
$$

### Parâmetros Físicos Utilizados

| Parâmetro | Descrição | Valor |
| :--- | :--- | :--- |
| $\Lambda$ | Constante Cosmológica | -0.001 |
| $L$ | Domínio do Fator de Escala ($a$) | 3.0 |
| $V_0$ | Profundidade do Poço | -1.0 (Caso estudado na Tabela 11) |

---

## 2. Objetivo e Metodologia

O projeto visa validar a capacidade das PINNs em resolver problemas espectrais multiestados, focando em:

* **Treinamento Autônomo:** Deixar a rede neural aprender **sozinha** as autofunções $\eta_n(a)$ e os autovalores $E_n$.
* [cite_start]**Restrições Quânticas:** Impor restrições de **ortogonalidade** ($\mathcal{L}_{\text{orto}}$) e **ordenação espectral** ($\mathcal{L}_{\text{ordenação}}$) para garantir a validade física das soluções[cite: 106, 107].
* **Comparação:** Comparar os autovalores obtidos pela PINN com os valores de referência da **Tabela 11** da dissertação (Método Espectral de Galerkin).

---

## 3. Estrutura e Componentes do Projeto

A implementação é organizada da seguinte forma:

| Diretório/Arquivo | Função Principal | Detalhe |
| :--- | :--- | :--- |
| `src/models/dnn.py` | Implementação da classe `SpectrumSolver`. | [cite_start]Contém a **MLP Densa** e a lógica para o cálculo do resíduo da EDP e a imposição das CC via transformação $\eta_{n}(a)=a(L-a)\eta_{n}^{raw}(a)$[cite: 97]. |
| `src/utils/helpers.py` | Funções Auxiliares. | Contém o cálculo do potencial $V(a)$, *plotting* das curvas de Loss e das autofunções, e gerenciamento de *checkpoints*. |
| `config.yaml` | **Hiperparâmetros de Treino/Física.** | Define `learning_rate`, `epochs`, `colocation_points` (2000), e os pesos da Loss ($\lambda_{orto}$, $\lambda_{ord}$). |
| `train_dl.py` | Script de Execução. | Executa o loop de treino, salvando o `best_pinns_model.pt` com base no menor **Loss EDP**. |

### Configurações de Treinamento

| Parâmetro | Valor |
| :--- | :--- |
| `num_states` | 15 |
| `learning_rate` | 0.0003 |
| `epochs` | 20000 |
| `colocation_points` | 2000 |
| `weight_ortogonalidade` | 500.0 |
| `weight_ordenação` | 50.0 |

---

## 4. Resultados Qualitativos e Limitações

A análise da **curva de Loss** (Figura 1) e das autofunções (Figura 2) fornece a seguinte conclusão:

* [cite_start]**Concordância Qualitativa:** As autofunções obtidas pelas PINNs demonstram **excelente concordância qualitativa** com as referências, exibindo o número correto de *nodos* (zeros) esperado pela teoria espectral[cite: 125, 126].
* **Instabilidade Numérica:** A curva de Loss (Loss EDP) frequentemente exibe **picos violentos e recorrentes** , indicando que, apesar das otimizações, a convergência foi marcada por **instabilidade numérica** (devido, em parte, à alta complexidade do problema multiestado e, historicamente, a uma alta taxa de aprendizado).
* **Robustez:** A robustez do modelo é garantida pelas perdas de Ortogonalidade e Condição de Contorno, que permanecem estáveis em $\sim 10^{-6}$, permitindo que o modelo extraia autovalores aceitáveis apesar da instabilidade da Loss EDP.

---

## 5. Conclusão Física (Eliminação da Singularidade)

[cite_start]A **Figura 3 (Pacote de Ondas)** confirma a principal implicação física do tratamento quântico: a evolução do pacote de ondas mostra que o fator de escala do universo $a$ **nunca se anula**[cite: 167]. [cite_start]As soluções são regulares (tipo buraco de minhoca) e eliminam a singularidade do Big Bang presente na dinâmica clássica[cite: 9, 179].

---

## Referências

* Monteiro, António. Cosmologia quântica computacional : aplicação do método espectral de Galerkin no estudo da dinâmica do universo primitivo descrito por radiação, constante cosmológica negativa e um potencial de Pöschl-Teller / António Monteiro. - 2025. 74 f. [cite_start]: il. [cite: 287]
* (Outras referências mencionadas no artigo original são citadas internamente.)
