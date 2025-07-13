.. Finite Automata documentation master file, created by
   sphinx-quickstart on Sun Jul 13 12:33:53 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Finite Automata Documentation
=============================

A Python library for working with finite automata.

This library provides tools for creating, manipulating, and minimizing both **Deterministic Finite Automata (DFA)** and **Nondeterministic Finite Automata (NFA)**.  
It includes an implementation of the **Kameda-Weiner algorithm** for NFA minimization, along with functionalities for conversion, reversal, and equivalence checking of automata.  
You can also convert any given NFA to a non-atomic or atomic NFA. Additionally, it integrates with `graphviz` to visualize the state diagrams of these automata.

Features
--------

* **DFA and NFA Implementation**  
  Create and modify DFAs and NFAs with an intuitive and flexible API.

* **Automata Operations**
  
  - Convert NFAs to DFAs using subset construction.
  - Minimize DFAs using the table-filling algorithm.
  - Reverse the language of an automaton.
  - Check for isomorphism between two DFAs.
  - Determine if two NFAs are equivalent.

* **Kameda-Weiner NFA Minimization**  
  As far as we know, this is the first clear and Pythonic implementation, with full verbose output of steps.

* **Visualization**  
  Generate clear and readable diagrams of your automata using `graphviz`.  
  For Kameda-Weiner, you can render the steps for minimization as a website or LaTeX document.

Installation
------------

To use this library, you'll need Python installed, as well as the `graphviz` and `requests` libraries:

.. code-block:: bash

   pip install graphviz requests

You will also need to install the Graphviz software on your system.  
Refer to the official `Graphviz download page <https://graphviz.org/download/>`_ for installation instructions based on your operating system.

Once dependencies are in place, you can clone the repository:

.. code-block:: bash

   git clone https://github.com/jsmaskeen/finite-automata.git

Usage
-----

See the ``Examples`` folder for detailed usage.

You can also run examples directly in Google Colab:

**1. Atomicity Function Checks**

.. raw:: html

   <a href="https://colab.research.google.com/github/jsmaskeen/finite-automata/blob/main/Examples/atomicity.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

**2. NFA - DFA Construction Examples**

.. raw:: html

   <a href="https://colab.research.google.com/github/jsmaskeen/finite-automata/blob/main/Examples/nfa-dfa_construction.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>

**3. Kameda-Weiner Minimisation and Enumeration of NFAs**

.. raw:: html

   <a href="https://colab.research.google.com/github/jsmaskeen/finite-automata/blob/main/Examples/kameda-weiner.ipynb" target="_blank">
     <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
   </a>
   
References
----------

- Brzozowski, J. A., & Tamm, H. (2013). *Minimal Nondeterministic Finite Automata and Atoms of Regular Languages*.  
  arXiv preprint arXiv:1301.5585.  
  https://arxiv.org/abs/1301.5585

- Brzozowski, J. A., & Tamm, H. (2011). *Theory of Atomata*.  
  arXiv preprint arXiv:1102.3901.  
  https://arxiv.org/abs/1102.3901

- Tsyganov, A. V. (2012). *Local Search Heuristics for NFA State Minimization Problem*.  
  International Journal of Communications, Network and System Sciences, 5(9), 638-643.  
  https://doi.org/10.4236/ijcns.2012.529074

- Kameda, T., & Weiner, P. (1970). *On the State Minimization of Nondeterministic Finite Automata*.  
  IEEE Transactions on Computers, C-19(7), 617-627.  
  https://doi.org/10.1109/T-C.1970.222994

Table of Contents
=================
.. toctree::
   :maxdepth: 4
   :caption: Contents:

   src.algo
   src.custom_typing
   src.fa
   src.utils


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`