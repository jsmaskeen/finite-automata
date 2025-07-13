from src.custom_typing.custom_types import STATE,DeltaOut
from typing import Set, Dict, Optional, Tuple, TYPE_CHECKING,Union,Iterable,List,Literal,FrozenSet,overload
import collections


if TYPE_CHECKING:
    from src.fa.dfa import DFA
    
    

class NFA:
    """Nondeterministic Finite Automaton (NFA) representation and operations.

    Models an NFA with a set of states, an input alphabet (Sigma), an initial
    subset of states, a set of final (accepting) states, and a transition
    relation (delta). Provides methods for state and transition management,
    reversal, conversion to an equivalent DFA (subset construction), and
    language equivalence checking via DFA minimization.
    """

    def __init__(
        self,
        *,
        States: Optional[Set[STATE]] = None,
        Sigma: Optional[Set[str]] = None,
        Initial: Optional[Set[STATE]] = None,
        Final: Optional[Set[STATE]] = None,
        delta: Optional[DeltaOut] = None,
    ) -> None:
        """Initialize an NFA with optional components.

        Args:
            States (Optional[Set[STATE]]):
                Initial set of states. Defaults to an empty set.
            Sigma (Optional[Set[str]]):
                Input alphabet. Defaults to an empty set.
            Initial (Optional[Set[STATE]]):
                Initial subset of states. Defaults to an empty set.
            Final (Optional[Set[STATE]]):
                Set of accepting states. Defaults to an empty set.
            delta (Optional[DeltaOut]):
                Transition mapping. Defaults to an empty DeltaOut.
        """
        self.States = States or set()
        self.Sigma = Sigma or set()
        self.Initial = Initial or set()
        self.Final = Final or set()
        self.delta = delta or DeltaOut()

    def addState(self, state: STATE):
        """Add a state to the NFA.

        Args:
            state (STATE):
                Identifier of the state to add.
        """
        self.States.add(state)

    def setSigma(self, sigma_set: Set[str]):
        """Define the NFA's input alphabet.

        Args:
            sigma_set (Set[str]):
                Set of symbols for the input alphabet.
        """
        self.Sigma = set(sigma_set)

    def setInitial(self, states: Union[STATE, Iterable[STATE]]) -> None:
        """Set the NFA's initial states.

        Converts a single state or an iterable of states into the internal set
        of initial states and ensures they are included in the overall state set.

        Args:
            states (Union[STATE, Iterable[STATE]]):
                A single state or collection of states to designate as initial.
        """
        if isinstance(states, (set, list, tuple, frozenset)):
            state_set: Set[STATE] = set(states)
        else:
            assert isinstance(states, STATE)
            state_set: Set[STATE] = {states}
        self.Initial = state_set
        self.States.update(self.Initial)

    def addFinal(self, state: STATE):
        """Mark a state as final (accepting).

        Args:
            state (STATE):
                The state to designate as accepting.
        """
        self.Final.add(state)
        self.addState(state)

    def addTransition(self, src: STATE, char: str, dest: STATE):
        """Add a transition to the NFA.

        Args:
            src (STATE):
                Source state for the transition.
            char (str):
                Input symbol triggering the transition.
            dest (STATE):
                Destination state reached on that symbol.
        """
        self.delta[src][char].add(dest)
        self.addState(src)
        self.addState(dest)

    def reversal(self):
        """Compute the reversal of the NFA.

        Constructs a new NFA that accepts the reversal of the original language
        by swapping initial and final states and reversing all transitions.

        Returns:
            NFA: The reversed NFA.
        """
        rev = NFA()
        rev.setSigma(self.Sigma)
        rev.setInitial(self.Final)
        for i in self.Initial:
            rev.addFinal(i)
        for src, transitions in self.delta.items():
            for char, destinations in transitions.items():
                for dest in destinations:
                    rev.addTransition(dest, char, src)
        for s in self.States:
            rev.addState(s)
        return rev

    @overload
    def toDFA(self, get_rev_dfa_map: Literal[False] = False) -> "DFA": ...
    @overload
    def toDFA(
        self, get_rev_dfa_map: Literal[True]
    ) -> Tuple["DFA", Dict[int, FrozenSet[STATE]]]: ...
    def toDFA(self, get_rev_dfa_map: bool = False):
        """Convert this NFA to an equivalent DFA using subset construction.

        Args:
            get_rev_dfa_map (bool, optional):
                If True, also return a mapping from DFA state IDs back to the
                corresponding NFA state subsets.

        Returns:
            DFA or Tuple[DFA, Dict[int, FrozenSet[STATE]]]:
                The constructed DFA, and optionally the reverse mapping if
                requested.
        """
        from src.fa.dfa import DFA
        dfa = DFA()
        dfa.setSigma(self.Sigma)
        rev_dfa_map: Optional[Dict[int, FrozenSet[STATE]]] = None

        if not self.Initial:
            if get_rev_dfa_map:
                rev_dfa_map = {}
                return dfa, rev_dfa_map
            else:
                return dfa

        initial_fs = frozenset(self.Initial)
        dfa_states_map = {initial_fs: 0}
        dfa.addState(0)
        if get_rev_dfa_map:
            rev_dfa_map = {0: initial_fs}
        dfa.setInitial(0)
        queue = collections.deque([initial_fs])
        next_state_id = 1
        sink_needed = False
        pending_sink_transitions: List[Tuple[int, str]] = []

        while queue:
            current_nfa_states = queue.popleft()
            current_dfa_id = dfa_states_map[current_nfa_states]
            if not self.Final.isdisjoint(current_nfa_states):
                dfa.addFinal(current_dfa_id)
            for char in self.Sigma:
                next_nfa_states: Set[STATE] = set()
                for nfa_state in current_nfa_states:
                    next_nfa_states.update(
                        self.delta.get(nfa_state, {}).get(char, set())
                    )
                if not next_nfa_states:
                    sink_needed = True
                    pending_sink_transitions.append((current_dfa_id, char))
                    continue
                next_fs = frozenset(next_nfa_states)
                if next_fs not in dfa_states_map:
                    dfa_states_map[next_fs] = next_state_id
                    dfa.addState(next_state_id)
                    if get_rev_dfa_map:
                        assert rev_dfa_map is not None
                        rev_dfa_map[next_state_id] = next_fs
                    queue.append(next_fs)
                    next_state_id += 1
                next_dfa_id = dfa_states_map[next_fs]
                dfa.addTransition(current_dfa_id, char, next_dfa_id)

        if sink_needed:
            sink_id = next_state_id
            dfa.addState(sink_id)
            for char in self.Sigma:
                dfa.addTransition(sink_id, char, sink_id)
            for current_dfa_id, char in pending_sink_transitions:
                dfa.addTransition(current_dfa_id, char, sink_id)
            if get_rev_dfa_map:
                assert rev_dfa_map is not None
                rev_dfa_map[sink_id] = frozenset()

        if get_rev_dfa_map:
            assert rev_dfa_map is not None
            return dfa, rev_dfa_map
        else:
            return dfa

    def is_equivalent_to(self, other_nfa: "NFA"):
        """Check language equivalence with another NFA.

        Converts both NFAs to minimized DFAs and tests for isomorphism.

        Args:
            other_nfa (NFA):
                The other NFA to compare against.

        Returns:
            bool: True if the two NFAs accept the same language, False otherwise.
        """
        min_dfa1 = self.toDFA().minimal()
        min_dfa2 = other_nfa.toDFA().minimal()
        return min_dfa1.is_isomorphic_to(min_dfa2)

    def __repr__(self):
        """Return a string representation of the NFA.

        Includes states, alphabet, initial and final states, and transitions.

        Returns:
            str: A multi-line description of the NFA.
        """
        return (
            f"NFA(\n"
            f"  States: {self.States}\n"
            f"  Alphabet: {self.Sigma}\n"
            f"  Initial States: {self.Initial}\n"
            f"  Final States: {self.Final}\n"
            f"  Transitions: \n{self.delta}\n)"
        )
