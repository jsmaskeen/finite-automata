from src.fa.nfa import NFA
from src.fa.dfa import DFA


def check_atomicity(nfa: NFA):
    """
    Check whether the given NFA is atomic.

    An NFA is atomic if its reversal, determinized, and minimized form has the same number
    of states before and after minimization. This property indicates that the NFA's reverse-
    deterministic structure is already minimal.

    Args:
        nfa (NFA): The nondeterministic finite automaton to check for atomicity.

    Returns:
        bool: True if the NFA is atomic, False otherwise.
    """
    rev = nfa.reversal()
    dfa_rev = rev.toDFA()
    dfa_min = dfa_rev.minimal()
    before = len(dfa_rev.States)
    after  = len(dfa_min.States)
    return before == after

def make_nonminimal(nfa: NFA) -> NFA:
    """
    Construct a new NFA which recognizes the same language as input NFA
    but this new NFA's subset construction yields a DFA which is ensured
    to be non-minimal by cloning each state.

    This function takes the input NFA and creates a new one with two copies of each original state:
    one with its original name and one with a "'" suffix. Transitions are updated so that every
    transition from a state p to q in the original NFA is mirrored to both q and q' in the new NFA.

    Args:
        nfa (NFA): The source NFA to clone into a non-minimal NFA.

    Returns:
        NFA: A new nondeterministic finite automaton with duplicated states and transitions,
             ensuring it's DFA (from subset construction) is non-minimal.
    """
    suc = NFA()
    suc.setSigma(nfa.Sigma)
    orig_states = list(nfa.States)
    
    state_map = {q: f"{i}" for i, q in enumerate(orig_states)}  
    clone_map = {q: f"{i}'" for i, q in enumerate(orig_states)}  
    
    all_new_states = set(state_map.values()) | set(clone_map.values())
    for s in all_new_states:
        suc.addState(s)
    
    initial_state = next(iter(nfa.Initial))
    suc.setInitial(state_map[initial_state])
    
    for f in nfa.Final:
        suc.addFinal(state_map[f])
        suc.addFinal(clone_map[f])
    
    for p in orig_states:
        for a in nfa.Sigma:
            for q in nfa.delta.get(p, {}).get(a, set()):
                suc.addTransition(state_map[p], a, state_map[q])  # p -> q
                suc.addTransition(state_map[p], a, clone_map[q])  # p -> q'
                suc.addTransition(clone_map[p], a, clone_map[q])  # p' -> q'
    return suc

def make_atomic_by_reverse_min(nfa: NFA) -> NFA:
    """
    Create an atomic NFA by reversing and minimizing.

    This function determinizes and minimizes the input NFA to a DFA, then reverses that DFA
    to obtain an NFA that is guaranteed to be atomic.

    Args:
        nfa (NFA): The original nondeterministic finite automaton.

    Returns:
        NFA: An atomic NFA obtained by reverse-minimization of the original NFA.
    """
    dfa_min:DFA = nfa.toDFA().minimal()
    atomic_nfa: NFA = dfa_min.reversal()
    return atomic_nfa