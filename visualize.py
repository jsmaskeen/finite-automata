import graphviz
import collections
from fa import NFA,DFA
from typing import Union

class AutomataVisualizer:
    @staticmethod
    def to_graphviz(automaton:Union[NFA,DFA]):
        dot = graphviz.Digraph(comment='Finite Automaton')
        dot.attr(rankdir='LR')
        dot.node('!!start_state!!', shape='none', label='')

        state_names = {s: str(s) for s in automaton.States}

        for state, name in state_names.items():
            if state in automaton.Final:
                dot.node(name, shape='doublecircle')
            else:
                dot.node(name, shape='circle')

        initial_states = automaton.Initial
        if not isinstance(initial_states, (set, frozenset, list)):
            initial_states = {initial_states}
        
        for state in initial_states:
            if state in state_names:
                dot.edge('!!start_state!!', state_names[state])

       
        transitions = collections.defaultdict(list)
        if isinstance(automaton, DFA): 
            for (src, char), dest in automaton.delta.items():
                transitions[(state_names[src], state_names[dest])].append(char)
        elif isinstance(automaton, NFA): 
            for src, trans_dict in automaton.delta.items():
                for char, dest_set in trans_dict.items():
                    for dest in dest_set:
                        transitions[(state_names[src], state_names[dest])].append(char)
        else:
            raise NotImplementedError
        for (src_name, dest_name), labels in transitions.items():
            dot.edge(src_name, dest_name, label=','.join(sorted(labels)))

        return dot
    