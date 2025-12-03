
# Projeto Final — Deep Learning: Aplicação de Redes Neurais Informadas pela Física (PINNs) na Solução da Equação de Wheeler–DeWitt em um Modelo quântico-cosmológico

**Autor:** António Monteiro
**Matrícula:** [DO2520057]
**Curso:** Doutorado em Modelagem Computacional, UERJ/IPRJ (Nova Friburgo)

---

## DescriÇão

Este projeto implementa **Redes Neurais Informadas pela Física (PINNs)** para resolver o problema de autovalores associado à **equação de Wheeler–DeWitt estacionária** de um modelo cosmológico quântico, comparando os resultados com os valores de referência obtidos numericamente (método espectral) na **Tabela 11** da dissertação.

A ideia central é:

> Em vez de resolver o problema de autovalores com um método numérico tradicional, deixar o **PINN aprender simultaneamente as autofunções \(\eta_n(a)\) e os autovalores \(E_n\)**, apenas impondo a equação diferencial, as condições de contorno e condições de ortogonalidade/ordenação.

---

## 1. Modelo físico

Consideramos a equação de Wheeler–DeWitt (dependente do tempo) do modelo cosmológico-quântico adotado na dissertação. Para soluções estacionárias do tipo

\[
\Psi(a,\tau) = e^{-iE\tau}\,\eta(a),
\]

obtemos o problema de autovalor:

\[
-\frac{d^2\eta}{da^2} + V(a)\,\eta(a) = 12E\,\eta(a), \quad a\in[0,L],
\]

com condições de contorno de Dirichlet homogêneas:

\[
\eta(0) = \eta(L) = 0.
\]

O potencial efetivo é dado por

\[
V(a) = 36a^2 + 12|\Lambda|a^4 + 12 a V_0 \,\text{sech}^2(a),
\]

onde:

- \(\Lambda\) é a constante cosmológica (aqui, \(\Lambda = -0{,}001\)),
- \(L = 3\) é o tamanho do domínio em \(a\),
- \(V_0\) controla a profundidade do poço (valores típicos: \(-1, -5, -10, -15\)).

Os **autovalores de referência** \(E_n\) para os primeiros 15 níveis de energia são aqueles apresentados na **Tabela 11 da dissertação**, para \(\Lambda=-0{,}001\) e \(L=3\).

---

## 2. Objetivo do projeto

- Utilizar **PINNs** para resolver o problema de autovalores da equação de Wheeler–DeWitt estacionária.
- Deixar a rede neural:
  - aprender **sozinha** as autofunções \(\eta_n(a)\),
  - e **aprender sozinha** os autovalores \(E_n\), que são tratados como **parâmetros treináveis**.
- Comparar os autovalores obtidos pela PINN com os valores da **Tabela 11**:
  - para diferentes valores de \(V_0\),
  - e para diferentes números de níveis considerados (\(5, 10, 15\)).

---

## 3. Estrutura do projeto

```text
.
├── config.yaml                 # Configurações do modelo, física e treino
├── requirements.txt            # Dependências principais do projeto
├── src
│   ├── models
│   │   └── dnn.py             # Implementação do PINN (SpectrumSolver)
│   └── utils
│       └── helpers.py         # Funções auxiliares (potencial, gráficos, etc.)
├── results
│   ├── best_pinns_model.pt    # Melhor modelo salvo durante o treino
│   ├── metrics_pinns.json     # Métricas numéricas (autovalores e erros)
│   └── figures
│       ├── loss_curve.png     # Curva da função de loss ao longo do treino
│       └── autofunctions_*.png# Gráficos das autofunções
└── main.py (opcional)         # Script para chamar o treino/avaliação


## Referências

Monteiro, António.
Cosmologia quântica computacional : aplicação do método
espectral de Galerkin no estudo da dinâmica do universo primitivo
descrito por radiação, constante cosmológica negativa e um
potencial de Pöschl-Teller / António Monteiro. - 2025.
74 f. : il.
