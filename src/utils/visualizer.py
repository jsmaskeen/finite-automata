from graphviz import Digraph  # type: ignore
import collections
from typing import Union
from src.fa.nfa import NFA
from src.fa.dfa import DFA
import json, requests
import base64
from urllib.parse import urlparse
from typing import DefaultDict, Tuple, List, Dict, Any,FrozenSet
from src.custom_typing.custom_types import StatesMap
import os

class Visualizer:

    @staticmethod
    def to_graphviz(automaton: Union[NFA, DFA]):
        """
        Convert an NFA or DFA into a Graphviz Digraph for visualization.

        Args:
            automaton (Union[NFA, DFA]): The finite automaton to visualize.

        Returns:
            Digraph: A Graphviz object representing the automaton.
        """
        dot = Digraph(comment="Finite Automaton")
        dot.attr(rankdir="LR") # type: ignore
        dot.node("!!start_state!!", shape="none", label="") # type: ignore

        state_names = {s: str(s) for s in automaton.States}

        for state, name in state_names.items():
            if state in automaton.Final:
                dot.node(name, shape="doublecircle") # type: ignore
            else:
                dot.node(name, shape="circle") # type: ignore

        initial_states = automaton.Initial
        if not isinstance(initial_states, (set, frozenset, list)):
            initial_states = {initial_states}

        for state in initial_states:
            if state in state_names:
                dot.edge("!!start_state!!", state_names[state]) # type: ignore

        transitions: DefaultDict[Tuple[str, str], List[str]] = collections.defaultdict(
            list
        )
        if isinstance(automaton, DFA):
            for (src, char), dest in automaton.delta.items():
                transitions[(state_names[src], state_names[dest])].append(char)
        else:
            for src, trans_dict in automaton.delta.items():
                for char, dest_set in trans_dict.items():
                    for dest in dest_set:
                        transitions[(state_names[src], state_names[dest])].append(char)
        for (src_name, dest_name), labels in transitions.items():
            dot.edge(src_name, dest_name, label=",".join(sorted(labels))) # type: ignore

        return dot

    @staticmethod
    def to_website(steps: Dict[str, Any]):
        """
        Convert the steps of Kameda-Weiner minimization process, and display it on the website, with graphviz renders.
        """
        
        def serialize_set(fs: FrozenSet[Union[int,str]]):
            return sorted(list(fs), key=str)

        def serialize_dfa(dfa: DFA) -> Dict[str, Any]:
            return {
                'type': 'DFA',
                'states': [str(s) for s in dfa.States],
                'alphabet': list(dfa.Sigma),
                'initial_state': str(dfa.Initial) if dfa.Initial else None,
                'final_states': [str(s) for s in dfa.Final],
                'transitions': {f"{src}_{sym}": str(dest) for (src, sym), dest in dfa.delta.items()},
                'dot': Visualizer.to_graphviz(dfa).source  
            }

        def serialize_nfa(nfa: NFA) -> Dict[str, Any]:
            return {
                'type': 'NFA',
                'states': [str(s) for s in nfa.States],
                'alphabet': list(nfa.Sigma),
                'initial_states': [str(s) for s in nfa.Initial],
                'final_states': [str(s) for s in nfa.Final],
                'transitions': {
                    str(src): {sym: [str(d) for d in dests] for sym, dests in trans.items()}
                    for src, trans in nfa.delta.items()
                },
                'dot': Visualizer.to_graphviz(nfa).source  
            }

        def serialize_states_map(sm: StatesMap) -> Dict[str, Any]:
            return {
                'rows': [serialize_set(r) for r in sm['rows']],
                'cols': [serialize_set(c) for c in sm['cols']],
                'matrix': [
                    [serialize_set(cell) if cell is not None else None for cell in row]
                    for row in sm['matrix']
                ]
            }

        def serialize(steps: Dict[str, Any]):
            """
            Serialize the Kameda-Weiner steps dictionary into a JSON string with Graphviz dot representations.

            Args:
                steps (Dict[str, Any]): The steps dictionary from KamedaWeinerMinimize.

            Returns:
                URL-safe base64 encoded JSON string.
            """
            serialized = {}

            if "Original NFA" in steps:
                serialized["Original NFA"] = serialize_nfa(steps["Original NFA"])

            if "Reverse NFA" in steps:
                serialized["Reverse NFA"] = serialize_nfa(steps["Reverse NFA"])

            if steps["DFA"]:
                dfa, dfa_map = steps["DFA"]
                serialized["DFA"] = {
                    "dfa": serialize_dfa(dfa),
                    "mapping": {str(k): serialize_set(v) for k, v in dfa_map.items()},
                }

            if steps["Dual of DFA"]:
                dual_dfa, dual_map = steps["Dual of DFA"]
                serialized["Dual of DFA"] = {
                    "dfa": serialize_dfa(dual_dfa),
                    "mapping": {str(k): serialize_set(v) for k, v in dual_map.items()},
                }

            if steps["States Map Reduction"]:
                serialized["States Map Reduction"] = [
                    serialize_states_map(sm) for sm in steps["States Map Reduction"]
                ]
                
            if steps["Maximal Prime Grids"]:
                serialized["Maximal Prime Grids"] = [
                    (serialize_set(rows), serialize_set(cols))
                    for rows, cols in steps["Maximal Prime Grids"]
                ]

            if steps["Cover Enumeration"]:
                serialized["Cover Enumeration"] = [
                    [(serialize_set(rows), serialize_set(cols)) for rows, cols in cover]
                    for cover in steps["Cover Enumeration"]
                ]

            if steps["Intersection Rule Result"]:
                serialized["Intersection Rule Result"] = [
                    serialize_nfa(nfa) for nfa in steps["Intersection Rule Result"]
                ]
                serialized["Legitimacy"] = steps["Legitimacy"]

            json_str = json.dumps(serialized)
            base64_str = base64.urlsafe_b64encode(json_str.encode()).decode()
            return base64_str

        def upload(text: str):
            response = requests.post("https://paste.rs", data=text.encode("utf-8"))
            if response.ok:
                return response.text.strip()
            else:
                raise Exception("Upload failed:", response.text)

        base64_data = serialize(steps)
        link = upload(base64_data)
        parsed = urlparse(link)
        last_segment = parsed.path.lstrip("/")
        return f"https://jsmaskeen.github.io/finite-automata/steps.html?data={last_segment}"
    
    @staticmethod
    def to_latex(steps: Dict[str, Any], dirname: str):
        """
        Generate a LaTeX file 'story.tex' representing the Kameda-Weiner minimization process steps.

        Args:
            steps (Dict[str, Any]): The steps dictionary from KamedaWeinerMinimize.
            dirname (str): The directory name where to store the tex file and dot objects.
        """
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        def set_to_latex(s: FrozenSet[Union[int, str]]) -> str:
            if not s:
                return r"$\emptyset$"
            elements = sorted([str(e) for e in s])
            # Use \text{} for non-numeric elements to handle state names like s0, s1
            formatted_elements = [
                e if e.isdigit() else r"\text{" + e + r"}" for e in elements
            ]
            return r"$\{" + ", ".join(formatted_elements) + r"\}$"

        def states_map_to_latex(sm: StatesMap) -> str:
            rows = sm['rows']
            cols = sm['cols']
            matrix = sm['matrix']
            num_cols = len(cols)
            col_headers = " & ".join([set_to_latex(c) for c in cols])
            table_rows: List[str] = []
            for r, row in zip(rows, matrix):
                row_header = set_to_latex(r)
                cells = " & ".join([set_to_latex(cell) if cell is not None else r"$\emptyset$" for cell in row])
                table_rows.append(row_header + " & " + cells + r" \\")
            table = (
                r"\resizebox{\textwidth}{!}{" + "\n" +
                r"\begin{tabular}{|c|" + "c|" * num_cols + r"}" + "\n" +
                r"\hline" + "\n" +
                r"\multicolumn{1}{|c|}{} & " + col_headers + r" \\" + "\n" +
                r"\hline" + "\n" +
                "\n".join(table_rows) + "\n" +
                r"\hline" + "\n" +
                r"\end{tabular}" + "\n" +
                r"}"
            )
            return table

        latex_code = [
            r"\documentclass{article}",
            r"\usepackage{graphicx}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",  # For \emptyset
            r"\usepackage{geometry}",
            r"\geometry{a4paper, margin=1in}",
            r"\begin{document}",
        ]

        if "Original NFA" in steps:
            nfa = steps["Original NFA"]
            dot = Visualizer.to_graphviz(nfa)
            dot_file = os.path.join(dirname, "original_nfa.dot")
            dot.save(dot_file) # type: ignore
            latex_code.append(r"\section{Original NFA}")
            latex_code.append(r"\resizebox{\textwidth}{!}{\includegraphics{" + os.path.basename(dot_file).replace(".dot", ".pdf") + r"}}")

        if "Reverse NFA" in steps:
            nfa = steps["Reverse NFA"]
            dot = Visualizer.to_graphviz(nfa)
            dot_file = os.path.join(dirname, "reverse_nfa.dot")
            dot.save(dot_file) # type: ignore
            latex_code.append(r"\section{Reverse NFA}")
            latex_code.append(r"\resizebox{\textwidth}{!}{\includegraphics{" + os.path.basename(dot_file).replace(".dot", ".pdf") + r"}}")

        if "DFA" in steps:
            dfa, _ = steps["DFA"]
            dot = Visualizer.to_graphviz(dfa)
            dot_file = os.path.join(dirname, "dfa.dot")
            dot.save(dot_file) # type: ignore
            latex_code.append(r"\section{DFA}")
            latex_code.append(r"\resizebox{\textwidth}{!}{\includegraphics{" + os.path.basename(dot_file).replace(".dot", ".pdf") + r"}}")

        if "Dual of DFA" in steps:
            dual_dfa, _ = steps["Dual of DFA"]
            dot = Visualizer.to_graphviz(dual_dfa)
            dot_file = os.path.join(dirname, "dual_dfa.dot")
            dot.save(dot_file) # type: ignore
            latex_code.append(r"\section{Dual of DFA}")
            latex_code.append(r"\resizebox{\textwidth}{!}{\includegraphics{" + os.path.basename(dot_file).replace(".dot", ".pdf") + r"}}")

        if "States Map Reduction" in steps:
            latex_code.append(r"\section{States Map Reduction}")
            for i, sm in enumerate(steps["States Map Reduction"], 1):
                latex_code.append(r"\subsection{States Map " + str(i) + "}")
                table = states_map_to_latex(sm)
                latex_code.append(table)

        if "Maximal Prime Grids" in steps:
            latex_code.append(r"\section{Maximal Prime Grids}")
            latex_code.append(r"\begin{itemize}")
            for rows, cols in steps["Maximal Prime Grids"]:
                rows_str = set_to_latex(rows)
                cols_str = set_to_latex(cols)
                latex_code.append(r"\item Rows: " + rows_str + ", Cols: " + cols_str)
            latex_code.append(r"\end{itemize}")

        if "Cover Enumeration" in steps:
            latex_code.append(r"\section{Cover Enumeration}")
            for i, cover in enumerate(steps["Cover Enumeration"], 1):
                latex_code.append(r"\subsection{Cover " + str(i) + "}")
                latex_code.append(r"\begin{itemize}")
                for rows, cols in cover:
                    rows_str = set_to_latex(rows)
                    cols_str = set_to_latex(cols)
                    latex_code.append(r"\item Rows: " + rows_str + ", Cols: " + cols_str)
                latex_code.append(r"\end{itemize}")

        if "Intersection Rule Result" in steps:
            latex_code.append(r"\section{Intersection Rule Result}")
            for i, (nfa, is_legitimate) in enumerate(zip(steps["Intersection Rule Result"], steps.get("Legitimacy", [])), 1):
                dot = Visualizer.to_graphviz(nfa)
                dot_file = os.path.join(dirname, f"intersection_nfa_{i}.dot")
                dot.save(dot_file) # type: ignore
                latex_code.append(r"\subsection{NFA " + str(i) + "}")
                latex_code.append(r"\resizebox{\textwidth}{!}{\includegraphics{" + os.path.basename(dot_file).replace(".dot", ".pdf") + r"}}")
                legitimacy_text = "legitimate" if is_legitimate else "not legitimate"
                latex_code.append(r"This NFA is " + legitimacy_text + r".")

        latex_code.append(r"\end{document}")

        with open(os.path.join(dirname, "steps.tex"), "w") as f:
            f.write("\n".join(latex_code))