from typing import Set, Dict, Optional, Tuple, DefaultDict,List
from src.custom_typing.custom_types import STATE
from src.utils.helper import Helper
import collections

    

class DFA:
    """Deterministic Finite Automaton (DFA) representation and operations.

    This class models a DFA with a set of states, an input alphabet (Sigma),
    a single initial state, a set of final (accepting) states, and a transition
    function (delta). It provides methods for constructing the DFA, converting
    to an NFA, computing the language reversal, minimizing the DFA, and checking
    isomorphism with another DFA.
    """

    def __init__(self):
        """Initialize an empty DFA.

        Creates empty sets for states, alphabet, and final states, sets the initial
        state to None, and initializes an empty transition dictionary.
        """
        self.States: Set[STATE] = set()
        self.Sigma: Set[str] = set()
        self.Initial: Optional[STATE] = None
        self.Final: Set[STATE] = set()
        self.delta: Dict[Tuple[STATE, str], STATE] = {}

    def addState(self, state: STATE):
        """Add a new state to the DFA.

        Args:
            state (STATE):
                The identifier for the new state, either a string or integer.
        """
        self.States.add(state)

    def setSigma(self, sigma_set: Set[str]):
        """Define the input alphabet of the DFA.

        Args:
            sigma_set (Set[str]):
                A set of characters representing the DFA's input alphabet.
        """
        self.Sigma = set(sigma_set)

    def setInitial(self, state: STATE):
        """Set the initial (start) state of the DFA.

        Adds the state to the DFA if it does not already exist.

        Args:
            state (STATE):
                The state to designate as the initial state.
        """
        self.Initial = state
        self.addState(state)

    def addFinal(self, state: STATE):
        """Mark a state as a final (accepting) state.

        Adds the state to the DFA if it does not already exist.

        Args:
            state (STATE):
                The state to designate as accepting.
        """
        self.Final.add(state)
        self.addState(state)

    def addTransition(self, src: STATE, char: str, dest: STATE):
        """Add a transition for the DFA.

        Args:
            src (STATE):
                The source state for the transition.
            char (str):
                The input symbol that triggers the transition.
            dest (STATE):
                The destination state for the transition.
        """
        self.delta[(src, char)] = dest
        self.addState(src)
        self.addState(dest)

    def toNFA(self):
        """Convert this DFA into an equivalent NFA.

        Returns:
            NFA:
                A nondeterministic finite automaton representing the same language
                as this DFA.
        """
        from src.fa.nfa import NFA
        nfa = NFA()
        nfa.setSigma(self.Sigma)
        for s in self.States:
            nfa.addState(s)
        assert self.Initial is not None
        nfa.setInitial(self.Initial)
        for f in self.Final:
            nfa.addFinal(f)
        for (src, char), dest in self.delta.items():
            nfa.addTransition(src, char, dest)
        return nfa

    def reversal(self):
        """Construct the NFA which accepts the reverse of the language accepted by this DFA

        Returns:
            NFA:
                An NFA accepting the reversal of the language of this DFA.
        """
        from src.fa.nfa import NFA
        rev_nfa = NFA()
        rev_nfa.setSigma(self.Sigma)
        rev_nfa.setInitial(self.Final)
        assert self.Initial is not None
        rev_nfa.addFinal(self.Initial)

        for (src, char), dest in self.delta.items():
            rev_nfa.addTransition(dest, char, src)

        for s in self.States:
            rev_nfa.addState(s)

        return rev_nfa

    def minimal(self):
        """Compute and return the minimal equivalent DFA.

        Uses completion (adding a trap state), the table-filling algorithm to
        distinguish states, and union-find to merge indistinguishable states.

        Returns:
            DFA:
                A minimized DFA equivalent to the original.
        """
        if self.Initial is None:
            return DFA()

        completed_states = self.States.copy()
        completed_delta = self.delta.copy()

        trap_state = "trap_state"
        while trap_state in completed_states:
            trap_state += "_"

        is_incomplete = any(
            (s, char) not in completed_delta for s in self.States for char in self.Sigma
        )

        if is_incomplete:
            completed_states.add(trap_state)
            for s in self.States:
                for char in self.Sigma:
                    if (s, char) not in completed_delta:
                        completed_delta[(s, char)] = trap_state
            for char in self.Sigma:
                completed_delta[(trap_state, char)] = trap_state

        states = sorted(list(completed_states), key=str)
        distinguishable: Set[Tuple[STATE, STATE]] = set()

        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                p, q = states[i], states[j]
                is_p_final = p in self.Final
                is_q_final = q in self.Final
                if is_p_final != is_q_final:
                    _p, _q = tuple(sorted((p, q), key=str))
                    distinguishable.add((_p, _q))

        while True:
            newly_marked: Set[Tuple[STATE, STATE]] = set()
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    p, q = states[i], states[j]
                    if tuple(sorted((p, q), key=str)) in distinguishable:
                        continue

                    for char in self.Sigma:
                        next_p = completed_delta.get((p, char))
                        next_q = completed_delta.get((q, char))

                        if next_p != next_q:
                            if (
                                tuple(sorted((next_p, next_q), key=str))
                                in distinguishable
                            ):
                                _p, _q = tuple(sorted((p, q), key=str))
                                newly_marked.add((_p, _q))
                                break

            if not newly_marked:
                break
            distinguishable.update(newly_marked)

        all_pairs = {
            tuple(sorted((states[i], states[j]), key=str))
            for i in range(len(states))
            for j in range(i + 1, len(states))
        }
        indistinguishable_pairs = all_pairs - distinguishable

        parent = {s: s for s in completed_states}
        dsu = Helper.dsu(parent)
        for p, q in indistinguishable_pairs:
            dsu.union(p, q)

        partitions: DefaultDict[STATE, List[STATE]] = collections.defaultdict(list)
        for s in completed_states:
            partitions[dsu.find(s)].append(s)

        min_dfa = DFA()
        min_dfa.setSigma(self.Sigma)

        partition_map = {
            frozenset(v): f"q{i}" for i, v in enumerate(partitions.values())
        }
        state_to_partition_name = {
            s: partition_map[frozenset(partitions[dsu.find(s)])]
            for s in completed_states
        }

        for p_name in partition_map.values():
            min_dfa.addState(p_name)

        min_dfa.setInitial(state_to_partition_name[self.Initial])

        for f_partition_fs, p_name in partition_map.items():
            if not f_partition_fs.isdisjoint(self.Final):
                min_dfa.addFinal(p_name)

        for p_frozenset, p_name in partition_map.items():
            representative = next(iter(p_frozenset))
            for char in self.Sigma:
                next_state = completed_delta[(representative, char)]
                next_partition_name = state_to_partition_name[next_state]
                min_dfa.addTransition(p_name, char, next_partition_name)

        return min_dfa

    def is_isomorphic_to(self, other_dfa: "DFA"):
        """Check whether this DFA is isomorphic to another DFA.

        Two DFAs are isomorphic if there exists a renaming of states that
        makes their transition structures and accepting behavior identical.

        Args:
            other_dfa (DFA):
                The DFA to compare against.

        Returns:
            bool:
                True if the DFAs are isomorphic, False otherwise.
        """
        if (
            self.Sigma != other_dfa.Sigma
            or len(self.States) != len(other_dfa.States)
            or len(self.Final) != len(other_dfa.Final)
        ):
            return False

        # BFS time !!!
        queue = collections.deque([(self.Initial, other_dfa.Initial)])
        visited_map = {self.Initial: other_dfa.Initial}

        while queue:
            q1, q2 = queue.popleft()
            assert q1 is not None
            assert q2 is not None
            if (q1 in self.Final) != (q2 in other_dfa.Final):
                return False

            for char in sorted(list(self.Sigma)):
                next_q1 = self.delta.get((q1, char))
                next_q2 = other_dfa.delta.get((q2, char))

                if (next_q1 is None) != (next_q2 is None):
                    return False

                if next_q1 is not None:
                    if next_q1 in visited_map:
                        if visited_map[next_q1] != next_q2:
                            return False
                    else:
                        visited_map[next_q1] = next_q2
                        queue.append((next_q1, next_q2))
        return True

    def __repr__(self):
        """Return a string representation of the DFA.

        Includes states, alphabet, initial state, final states, and transitions.

        Returns:
            str:
                A multi-line representation of the DFA's components.
        """
        return (
            f"DFA(\n"
            f"  States: {self.States}\n"
            f"  Alphabet: {self.Sigma}\n"
            f"  Initial State: {self.Initial}\n"
            f"  Final States: {self.Final}\n"
            f"  Transitions: {self.delta}\n)"
        )
