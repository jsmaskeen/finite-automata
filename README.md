## Finite Automata

An Python library for working with finite automata.

This library provides tools for creating, manipulating, and minimizing both Deterministic Finite Automata (DFA) and Nondeterministic Finite Automata (NFA). 
It features the implementation of the Kameda-Weiner algorithm for NFA minimization, along with functionalities for conversion, reversal, and equivalence checking of automata. You can also convert any given NFA to non-atomic NFA or atomic NFA. Additionally, it integrates with `graphviz` to offer a simple way to visualize the state diagrams of these automata.

## Documentation:

1. [Read online](http://jsmaskeen.github.io/finite-automata)
2. [Read PDF](./finiteautomata.pdf) 

> Made using Sphinx

## Features 
  * **DFA and NFA Implementation**: Create and modify DFAs and NFAs with an intuitive and flexible API.
  * **Automata Operations**:
      * Convert NFAs to DFAs using subset construction.
      * Minimize DFAs with the table-filling algorithm.
      * Reverse the language of an automaton.
      * Check for isomorphism between two DFAs.
      * Determine if two NFAs are equivalent.
  * **Kameda-Weiner NFA Minimization**: AFAIK this si the first clear and Python implementation, with full verbose output of steps.
  * **Visualization**: Generate clear and readable diagrams of your automata using `graphviz`. Specifically for Kameda-Weiner, you can render the steps for minimization as a website, or LaTeX document.

-----

## Installation

To use this library, you'll need to have Python installed, as well as the `graphviz` and `requests` library. 
```bash
pip install graphviz requests
```
You will also need to install the Graphviz software on your system. Please refer to the official [Graphviz download page](https://graphviz.org/download/) for instructions for your operating system.

Once the dependencies are in place, you can clone this repository to your local machine:

```bash
git clone https://github.com/jsmaskeen/finite-automata.git
```

-----

## Usage

See the [Examples](./Examples) folder.
Alse see generated PDFs at [generated_pdf_examples](./generated_pdf_examples) folder.
You can run the examples directly from these buttons in Google Colab:

1. Atomicity Function checks

  <a href="https://colab.research.google.com/github/jsmaskeen/finite-automata/blob/main/Examples/atomicity.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

2. NFA - DFA Construction Examples

  <a href="https://colab.research.google.com/github/jsmaskeen/finite-automata/blob/main/Examples/nfa-dfa_construction.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>

3. Kameda-Weiner Minimisation and Enumeration of NFAs

  <a href="https://colab.research.google.com/github/jsmaskeen/finite-automata/blob/main/Examples/kameda-weiner.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>


-----

## References

- Brzozowski, J. A., & Tamm, H. (2013). **Minimal Nondeterministic Finite Automata and Atoms of Regular Languages**. *arXiv preprint arXiv:1301.5585*. Retrieved from [https://arxiv.org/abs/1301.5585](https://arxiv.org/abs/1301.5585) 

- Brzozowski, J. A., & Tamm, H. (2011). **Theory of Atomata**. *arXiv preprint arXiv:1102.3901*. Retrieved from [https://arxiv.org/abs/1102.3901](https://arxiv.org/abs/1102.3901) 

- Tsyganov, A. V. (2012). **Local Search Heuristics for NFA State Minimization Problem**. *International Journal of Communications, Network and System Sciences, 5*(9), 638–643. [https://doi.org/10.4236/ijcns.2012.529074](https://doi.org/10.4236/ijcns.2012.529074) 

- Kameda, T., & Weiner, P. (1970). **On the State Minimization of Nondeterministic Finite Automata**. *IEEE Transactions on Computers, C‑19*(7), 617–627. [https://doi.org/10.1109/T‑C.1970.222994](https://doi.org/10.1109/T‑C.1970.222994) 

