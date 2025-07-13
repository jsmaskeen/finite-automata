import collections
import itertools
from collections import deque
from typing import (
    Any,
    Union,
    Set,
    TypeAlias,
    Optional,
    Dict,
    Tuple,
    DefaultDict,
    List,
    Iterable,
    FrozenSet,
    overload,
    TypedDict,
    Literal,
)

STATE: TypeAlias = Union[str, int]


class StatesMap(TypedDict):
    """Mapping of DFA state partitions used in the Kameda-Weiner minimization algorithm.

    In the context of the Kameda-Weiner algorithm, the states map matrix records how
    groups of DFA states (rows and columns) intersect during minimization. This structure
    supports tracking which combined state sets remain distinguishable or mergeable.

    Attributes:
        rows (List[FrozenSet[int]]):
            Sequence of frozen sets, each representing a block of DFA state indices
            corresponding to a row partition in the states map.
        cols (List[FrozenSet[int]]):
            Sequence of frozen sets, each representing a block of DFA state indices
            corresponding to a column partition in the states map.
        matrix (List[List[Optional[FrozenSet[STATE]]]]):
            A two-dimensional list with dimensions len(rows) × len(cols). Each cell
            holds either:
              - A frozen set of STATE elements indicating the intersection of the
                respective row’s and column’s state blocks during minimization.
              - None if no intersection (i.e., the blocks do not share any states).
    """

    rows: List[FrozenSet[int]]
    cols: List[FrozenSet[int]]
    matrix: List[List[Optional[FrozenSet[STATE]]]]


class Helper:
    """Utility class providing helper algorithms and data structures."""

    def __init__(self) -> None:
        """Initialize the Helper instance."""
        pass
    @staticmethod
    def dsu(parent: dict[Any, Any]):
        """Construct a disjoint-set union (DSU) structure over the given parent mapping.

        The DSU supports efficient union and find operations with path compression
        on the provided parent dictionary.

        Args:
            parent (dict[Any, Any]):
                A mapping where each key is an element and each value is its current
                parent in the disjoint-set forest. Initially, each element should
                map to itself.

        Returns:
            DSU:
                An object exposing `find` and `union` methods for managing the
                disjoint-set structure.
        """

        class DSU:
            """Disjoint-set union (Union-Find) data structure with path compression."""

            def __init__(self, parent: dict[Any, Any]) -> None:
                """Initialize the DSU with a parent mapping.

                Args:
                    parent (dict[Any, Any]):
                        A mapping of each element to its parent in the set forest.
                """
                self.parent = parent

            def find(self, s: Any):
                """Find the representative (root) of the set containing `s`.

                This method applies path compression, updating the parent of `s`
                directly to the root for future efficiency.

                Args:
                    s (Any):
                        The element whose set representative is sought.

                Returns:
                    Any:
                        The root element representing the set containing `s`.
                """
                if self.parent[s] == s:
                    return s
                self.parent[s] = self.find(self.parent[s])
                return self.parent[s]

            def union(self, s1: Any, s2: Any):
                """Merge the sets containing `s1` and `s2`.

                If the two elements are in different sets, their roots are united
                by making one root point to the other.

                Args:
                    s1 (Any):
                        An element in the first set.
                    s2 (Any):
                        An element in the second set.
                """
                root1, root2 = self.find(s1), self.find(s2)
                if root1 != root2:
                    self.parent[root2] = root1

        return DSU(parent)
    
    @staticmethod
    def print_states_map(sm:StatesMap):
        """
        Helper function to pretty print a StatesMap object.
        
        Args:
            sm (StatesMap):
                The StatesMap to print.
        """
        def fmt(entry: Optional[FrozenSet[Union[int,STATE]]]) -> str:
            if entry is None:
                return '\u03C6'
            return '{' + ', '.join(sorted(map(str, entry))) + '}'
        
        rows = [fmt(r) for r in sm["rows"]]
        cols = [fmt(c) for c in sm["cols"]]
        mat  = sm["matrix"]
        pad:int = 4
        table = [
            [("\u03C6" if cell is None else fmt(cell)) for cell in row]
            for row in mat
        ]
        
        w0 = max(len(s) for s in rows + [""]) + pad
        
        col_widths:List[int] = []
        for j, ch in enumerate(cols):
            col_cells = [table[i][j] for i in range(len(rows))]
            wj = max(len(ch), *(len(c) for c in col_cells)) + pad
            col_widths.append(wj)

        print("".rjust(w0), end="")
        for j, ch in enumerate(cols):
            print(ch.rjust(col_widths[j]), end="")
        print()
        
        for i, row_label in enumerate(rows):
            print(row_label.rjust(w0), end="")
            for j in range(len(cols)):
                print(table[i][j].rjust(col_widths[j]), end="")
            print()


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


class DeltaIn(DefaultDict[str, Set[STATE]]):
    """Mapping from input symbols to sets of destination states for an NFA.

    This dictionary subclass maps each input symbol (string) to a set of
    NFA states (STATE) reachable on that symbol from a given source state.

    Example:
        delta_in = DeltaIn()
        delta_in['a'].add(state1)
        delta_in['b'].update({state2, state3})
    """

    def __init__(self):
        """Initialize an empty DeltaIn mapping.

        Each new key will default to an empty set of STATE.
        """
        super().__init__(set)
    
    def __repr__(self) -> str:
        """Returns a string representation of the DeltaIn mapping.

        Format:
            {'a': {s1, s2}, 'b': {s3}}

        Returns:
            str: A formatted string showing symbol-to-state-set mappings.
        """
        transitions = [f"'{sym}': {{{', '.join(map(str, sorted(dests)))}}}" for sym, dests in self.items()]
        return "{" + ", ".join(transitions) + "}"


class DeltaOut(DefaultDict[STATE, DeltaIn]):
    """Mapping from source states to their outgoing transitions in an NFA.

    This dictionary subclass maps each source state (STATE) to a DeltaIn
    instance, which in turn maps input symbols to sets of destination
    states. Together, DeltaOut[src][char] gives the set of states reachable
    from src on symbol char.

    Example:
        delta_out = DeltaOut()
        delta_out[src_state]['a'].add(dest_state)
        # Now dest_state is in delta_out[src_state]['a']
    """

    def __init__(self):
        """Initialize an empty DeltaOut mapping.

        Each new key will default to an empty DeltaIn for that source state.
        """
        super().__init__(DeltaIn)
        
    def __repr__(self) -> str:
        """Returns a string representation of the DeltaOut mapping.

        Format:
            s0 -> {'a': {s1}, 'b': {s2, s3}}
            s1 -> {'a': {s0}}

        Each line represents a source state and its transitions.

        Returns:
            str: A formatted multi-line string showing state transitions.
        """
        lines:List[str] = []
        for src in sorted(self.keys()):
            transitions = self[src]
            trans_str = ", ".join(f"'{sym}': {{{', '.join(map(str, sorted(dests)))}}}"
                                  for sym, dests in transitions.items())
            lines.append(f"{src} -> {{{trans_str}}}")
        return "\n".join(lines)


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
    def toDFA(self, get_rev_dfa_map: Literal[False] = False) -> DFA: ...
    @overload
    def toDFA(
        self, get_rev_dfa_map: Literal[True]
    ) -> Tuple[DFA, Dict[int, FrozenSet[STATE]]]: ...
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


class KamedaWeinerMinimize(NFA):
    """State minimization of an NFA using the Kameda-Weiner algorithm.

    Extends NFA to perform the Kameda-Weiner synthesis procedure, which
    finds a minimum-state NFA equivalent to the given NFA by analyzing
    the reduced automaton matrix (RAM), identifying prime grids, and
    applying the intersection rule over minimal covers.

    Inherits all standard NFA functionality and adds methods specific
    to the Kameda-Weiner minimization steps.
    """

    def __init__(self, nfa: NFA, store_progress: bool = False,verbose: bool = False):
        """Initialize the minimizer from an existing NFA.

        Copies the state set, alphabet, transition relation, initial
        and final states from the given NFA.

        Args:
            nfa (NFA):
                The nondeterministic finite automaton to minimize.
            record_progress (bool, optional):
                Set this to True if you want to store each step of
                Kameda-Weiner minimization process.
            verbose (bool, optional):
                Set this to True if you want to display each step of
                Kameda-Weiner minimization process.
        """
        super().__init__(
            States=nfa.States,
            Sigma=nfa.Sigma,
            delta=nfa.delta,
            Initial=nfa.Initial,
            Final=nfa.Final,
        )
        assert isinstance(nfa, NFA)
        self.store_progess: bool = store_progress
        self.verbose:bool = verbose
        self.steps: Dict[str, Any] = {
            "Original NFA":nfa,
            "Reverse NFA":None,
            "DFA": None,
            "Dual of DFA": None,
            "States Map Reduction": [],
            "Maximal Prime Grids": None,
            "Cover Enumeration": [],
            "Intersection Rule Result": [],
            "Legitimacy": [],
        }

    def construct_states_map(
        self,
        dfa_map: Dict[int, FrozenSet[STATE]],
        dfa_dual_map: Dict[int, FrozenSet[STATE]],
    ):
        """Build the initial states map (SM) matrix from DFA and dual DFA.

        Given the forward subset-construction mapping of the NFA to a DFA
        and the dual DFA mapping, construct a StatesMap that partitions
        NFA states into rows and columns and records their intersections.

        Args:
            dfa_map (Dict[int, FrozenSet[STATE]]):
                Mapping from DFA state ID to its corresponding NFA-state subset.
            dfa_dual_map (Dict[int, FrozenSet[STATE]]):
                Mapping from dual-DFA state ID to its corresponding NFA-state subset.

        Returns:
            StatesMap: The Initial States Map Matrix
        """
        num_rows = len(dfa_map)
        num_cols = len(dfa_dual_map)
        matrix: List[List[Optional[FrozenSet[STATE]]]] = [
            [None] * num_cols for _ in range(num_rows)
        ]

        for r in range(num_rows):
            for c in range(num_cols):
                intersection = dfa_map[r].intersection(dfa_dual_map[c])
                if intersection:
                    matrix[r][c] = intersection

        states_map: StatesMap = {
            "rows": [frozenset([i]) for i in range(num_rows)],
            "cols": [frozenset([i]) for i in range(num_cols)],
            "matrix": matrix,
        }

        return states_map

    def reduce_states_map(self, states_map: StatesMap):
        """Compute the reduced states map (RSM) by merging equivalent rows/columns.

        Iteratively merges rows with identical patterns of nonempty entries,
        then columns likewise, until no further merges are possible.

        Args:
            states_map (StatesMap):
                The initial or current states map to reduce.

        Returns:
            StatesMap: The reduced map with merged rows/columns.
        """
        current_map: StatesMap = states_map
        counter:int = 1
        if self.verbose:
            print("\nReducing States Map matrix\n")
        while True:
            if self.store_progess:
                self.steps["States Map Reduction"].append(current_map)
            if self.verbose:
                print(f"\nStep {counter}:")
                Helper.print_states_map(current_map)
                counter+=1
            
            row_patterns: DefaultDict[Tuple[bool, ...], List[int]] = (
                collections.defaultdict(list)
            )
            matrix: List[List[Optional[frozenset[STATE]]]] = current_map["matrix"]
            for i, row in enumerate(matrix):
                pattern: Tuple[bool, ...] = tuple(cell is not None for cell in row)
                row_patterns[pattern].append(i)

            merged = False
            for indices in row_patterns.values():
                if len(indices) > 1:
                    current_map = self.merge_rows(current_map, indices)
                    merged = True
                    break
            if merged:
                continue

            col_patterns: DefaultDict[Tuple[bool, ...], List[int]] = (
                collections.defaultdict(list)
            )
            for j in range(len(current_map["cols"])):
                pattern = tuple(
                    current_map["matrix"][i][j] is not None
                    for i in range(len(current_map["rows"]))
                )
                col_patterns[pattern].append(j)

            for indices in col_patterns.values():
                if len(indices) > 1:
                    current_map = self.merge_cols(current_map, indices)
                    merged = True
                    break
            if merged:
                continue

            break
        if self.store_progess:
                self.steps["States Map Reduction"].append(current_map)
        if self.verbose:
            print(f"\nStep {counter}:")
            Helper.print_states_map(current_map)
        return current_map

    def merge_rows(self, sm: StatesMap, indices_to_merge: List[int]):
        """Merge a set of rows in the states map into a single row.

        Combines the specified row indices by unioning their row-blocks
        and merging their matrix entries component-wise.

        Args:
            sm (StatesMap):
                The current states map.
            indices_to_merge (List[int]):
                List of row indices that should be merged.

        Returns:
            StatesMap: A new states map with those rows merged.
        """
        new_rows: List[FrozenSet[int]] = []
        new_matrix: List[List[Optional[frozenset[STATE]]]] = []
        empty: FrozenSet[int] = frozenset()
        merged_row_content = empty.union(*(sm["rows"][i] for i in indices_to_merge))
        first_idx = min(indices_to_merge)

        for i in range(len(sm["rows"])):
            if i in indices_to_merge:
                if i == first_idx:
                    new_rows.append(merged_row_content)
                    merged_matrix_row: List[Optional[FrozenSet[STATE]]] = [None] * len(
                        sm["cols"]
                    )
                    for col_idx in range(len(sm["cols"])):
                        empty2: Set[STATE] = set()
                        cells: List[FrozenSet[STATE]] = []
                        for row_idx in indices_to_merge:
                            cell = sm["matrix"][row_idx][col_idx]
                            if cell is not None:
                                cells.append(cell)
                        combined_cell = empty2.union(*cells)
                        if combined_cell:
                            merged_matrix_row[col_idx] = frozenset(combined_cell)
                    new_matrix.append(merged_matrix_row)
            else:
                new_rows.append(sm["rows"][i])
                new_matrix.append(sm["matrix"][i])

        return StatesMap(rows=new_rows, cols=sm["cols"], matrix=new_matrix)

    def merge_cols(self, sm: StatesMap, indices_to_merge: List[int]):
        """Merge a set of columns in the states map into a single column.

        Combines the specified column indices by unioning their column-blocks
        and merging their matrix entries component-wise.

        Args:
            sm (StatesMap):
                The current states map.
            indices_to_merge (List[int]):
                List of column indices that should be merged.

        Returns:
            StatesMap: A new states map with those columns merged.
        """
        new_cols: List[FrozenSet[int]] = []
        new_matrix: List[List[Optional[frozenset[STATE]]]] = []
        empty: FrozenSet[int] = frozenset()
        merged_col_content = empty.union(*(sm["cols"][j] for j in indices_to_merge))
        first_idx = min(indices_to_merge)
        indices_set = set(indices_to_merge)

        for j in range(len(sm["cols"])):
            if j in indices_set:
                if j == first_idx:
                    new_cols.append(merged_col_content)
            else:
                new_cols.append(sm["cols"][j])

        for i in range(len(sm["rows"])):
            new_row: List[Optional[FrozenSet[STATE]]] = []
            for j in range(len(sm["cols"])):
                if j == first_idx:
                    empty2: Set[STATE] = set()
                    cells: List[FrozenSet[STATE]] = []
                    for col_idx in indices_to_merge:
                        cell = sm["matrix"][i][col_idx]
                        if cell is not None:
                            cells.append(cell)
                    combined_cell = empty2.union(*cells)
                    new_row.append(frozenset(combined_cell) if combined_cell else None)
                elif j not in indices_set:
                    new_row.append(sm["matrix"][i][j])
            new_matrix.append(new_row)

        return StatesMap(rows=sm["rows"], cols=new_cols, matrix=new_matrix)

    def is_grid_prime(
        self, sm: StatesMap, grid_rows: FrozenSet[int], grid_cols: FrozenSet[int]
    ):
        """Check whether a given grid is prime (cannot be extended).

        A grid is prime if all intersections between the specified row set
        and column set in the states map are non-None, and it is not strictly
        contained in any larger grid.

        Args:
            sm (StatesMap):
                The reduced states map.
            grid_rows (FrozenSet[int]):
                The set of row indices defining the grid.
            grid_cols (FrozenSet[int]):
                The set of column indices defining the grid.

        Returns:
            bool: True if the grid is prime, False otherwise.
        """
        return all(sm["matrix"][r][c] is not None for r in grid_rows for c in grid_cols)

    def find_maximal_prime_grids(self, sm: StatesMap):
        """Enumerate all maximal prime grids in the reduced states map.

        Explores subgrids by breadth-first reduction of rows or columns
        containing None entries, and collects those grids that are prime
        and not strictly contained in any other prime grid.

        Args:
            sm (StatesMap):
                The reduced states map.

        Returns:
            List[Tuple[FrozenSet[int], FrozenSet[int]]]:
                List of pairs (rows, cols) each defining a maximal prime grid.
        """
        if self.verbose:
            print("\nFinding Maximal Prime Grids\n")
            
        num_rows, num_cols = len(sm["rows"]), len(sm["cols"])
        initial_grid: Tuple[FrozenSet[int], FrozenSet[int]] = (
            frozenset(range(num_rows)),
            frozenset(range(num_cols)),
        )
        q = deque([initial_grid])
        seen = {initial_grid}
        maximal_grids: Set[Tuple[FrozenSet[int], FrozenSet[int]]] = set()

        while q:
            rows, cols = q.popleft()
            if self.is_grid_prime(sm, rows, cols):
                if any(
                    rows.issubset(max_r) and cols.issubset(max_c)
                    for max_r, max_c in maximal_grids
                ):
                    continue
                maximal_grids = {
                    (mr, mc)
                    for mr, mc in maximal_grids
                    if not (mr.issubset(rows) and mc.issubset(cols))
                }
                maximal_grids.add((rows, cols))
            else:
                rows_with_zeros = {
                    r for r in rows if any(sm["matrix"][r][c] is None for c in cols)
                }
                cols_with_zeros = {
                    c for c in cols if any(sm["matrix"][r][c] is None for r in rows)
                }

                for r_to_remove in rows_with_zeros:
                    new_rows = rows - {r_to_remove}
                    if new_rows and (new_rows, cols) not in seen:
                        q.append((new_rows, cols))
                        seen.add((new_rows, cols))
                for c_to_remove in cols_with_zeros:
                    new_cols = cols - {c_to_remove}
                    if new_cols and (rows, new_cols) not in seen:
                        q.append((rows, new_cols))
                        seen.add((rows, new_cols))
        maxi_grids = list(maximal_grids)
        if self.store_progess:
            self.steps['Maximal Prime Grids'] = maxi_grids
        if self.verbose:
            print(f"Found Maximal Grids:\n{'\n'.join([ f'Rows:{set(i)}\nColumns:{set(j)}\n' for i,j in maxi_grids])}\n")
        return maxi_grids

    def is_cover(
        self, sm: StatesMap, grids: Tuple[Tuple[FrozenSet[int], FrozenSet[int]], ...]
    ):
        """Determine if a set of grids covers all 1-entries in the RAM.

        A cover is valid if every nonempty cell in the states map matrix
        lies within at least one of the supplied grids.

        Args:
            sm (StatesMap):
                The reduced states map.
            grids (Tuple[Tuple[FrozenSet[int], FrozenSet[int]], ...]):
                Sequence of grids to test as a cover.

        Returns:
            bool: True if the grids form a complete cover, False otherwise.
        """
        return all(
            any(r in gr and c in gc for gr, gc in grids)
            for r in range(len(sm["rows"]))
            for c in range(len(sm["cols"]))
            if sm["matrix"][r][c] is not None
        )

    def find_all_minimal_covers(
        self, sm: StatesMap, prime_grids: List[Tuple[FrozenSet[int], FrozenSet[int]]]
    ):
        """Generate all covers of minimal size from the prime grids.

        Yields each distinct minimal cover (list of grids) in increasing
        order of grid count.

        Args:
            sm (StatesMap):
                The reduced states map.
            prime_grids (List[Tuple[FrozenSet[int], FrozenSet[int]]]):
                Candidate prime grids.

        Yields:
            Iterator[List[Tuple[FrozenSet[int], FrozenSet[int]]]]:
                Each minimal cover as a list of grid coordinate pairs.
        """
        for k in range(1, len(self.States) + 1):
            for cover_candidate in itertools.combinations(prime_grids, k):
                if self.is_cover(sm, cover_candidate):
                    yield list(cover_candidate)
        return None

    def apply_intersection_rule(
        self,
        dfa: DFA,
        rsm: StatesMap,
        cover: List[Tuple[FrozenSet[int], FrozenSet[int]]],
    ):
        """Synthesize an NFA from a chosen cover via the intersection rule.

        Constructs a new NFA whose states correspond to the grids in the cover,
        using the DFA transition structure to determine transitions by
        intersecting preimages.

        Args:
            dfa (DFA): The forward-subset DFA of the original NFA.
            rsm (StatesMap): The reduced states map.
            cover (List[Tuple[FrozenSet[int], FrozenSet[int]]]):
                The selected minimal cover of prime grids.

        Returns:
            NFA: A synthesized NFA corresponding to this cover.
        """
        num_new_states = len(cover)

        f: Dict[STATE, Set[STATE]] = {
            dfa_state: {
                grid_idx for grid_idx, (gr, _) in enumerate(cover) if rsm_row_idx in gr
            }
            for rsm_row_idx, original_dfa_states in enumerate(rsm["rows"])
            for dfa_state in original_dfa_states
        }

        rev_f: List[FrozenSet[int]] = []
        for gr, _ in cover:
            empty: FrozenSet[int] = frozenset()
            rev_f.append(empty.union(*(rsm["rows"][rsm_row_idx] for rsm_row_idx in gr)))

        empty2: Set[STATE] = set()

        new_initial: Optional[Set[STATE]] = (
            f.get(dfa.Initial, empty2) if dfa.Initial is not None else None
        )
        assert new_initial is not None
        new_final = {
            s
            for s in range(num_new_states)
            if rev_f[s] and rev_f[s].issubset(dfa.Final)
        }

        min_nfa = NFA()
        min_nfa.setSigma(self.Sigma)
        for s in range(num_new_states):
            min_nfa.addState(s)
        min_nfa.setInitial(new_initial)
        for s in new_final:
            min_nfa.addFinal(s)

        for s1 in range(num_new_states):
            for letter in self.Sigma:
                for s2 in range(num_new_states):
                    if rev_f[s1] and all(
                        dfa.delta.get((orig, letter)) in rev_f[s2] for orig in rev_f[s1]
                    ):
                        min_nfa.addTransition(s1, letter, s2)
        return min_nfa

    @overload
    def run(self, get_all_nfas: Literal[False] = False) -> NFA: ...
    @overload
    def run(self, get_all_nfas: Literal[True]) -> List[Tuple[NFA, bool]]: ...
    def run(self, get_all_nfas: bool = False):
        """Execute the full Kameda-Weiner minimization procedure.

        1. Convert the NFA to a DFA and its dual.
        2. Build and reduce the states map.
        3. Find prime grids and enumerate minimal covers.
        4. Apply the intersection rule to each cover until a
           legitimate (language-preserving) NFA is found.

        Args:
            get_all_nfas (bool, optional):
                If True, return all candidate NFAs in non decreasing order
                of number of states with a flag indicating equivalence to
                the original. If False, return the first valid minimized
                NFA found.

        Returns:
            NFA or List[Tuple[NFA, bool]]:
                The minimized NFA, or a list of (NFA, is_equivalent) pairs.
        """
        dfa, dfa_map = self.toDFA(True)
        if self.store_progess:
            self.steps['DFA'] = (dfa,dfa_map)
            
        if self.verbose:
            print(f"Input NFA for some L:\n{self}\n")
            print(f"DFA for L:\n{dfa}\nMapping:\n{dfa_map}\n")
            
        nfa_rev = self.reversal()
        dual_dfa, dfa_dual_map = nfa_rev.toDFA(True)
        
        if self.store_progess:
            self.steps['Dual of DFA'] = (dual_dfa,dfa_dual_map)
            self.steps["Reverse NFA"] = nfa_rev
        
        if self.verbose:
            print(f"NFA for rev(L):\n{nfa_rev}\n")
            print(f"DFA for rev(L):\n{dual_dfa}\nMapping:\n{dfa_dual_map}\n")
            
        states_map = self.construct_states_map(dfa_map, dfa_dual_map)

        if self.store_progess:
            self.steps['States Map Reduction'].append(states_map)
        
        if self.verbose:
            Helper.print_states_map(states_map)
        
        rsm = self.reduce_states_map(states_map)
        if self.verbose:
            print("\nSuccessfully reduced the States Map matrix. You can see above the RSM.\n")
        prime_grids = self.find_maximal_prime_grids(rsm)
        
        all_min_covers = self.find_all_minimal_covers(rsm, prime_grids)
        
        if not all_min_covers:
            if self.verbose:
                print("\nNo minimal cover found! Given NFA cannot be reduced using Kameda-Weiner\n")
            return self
        miiini = None
        all_nfas: List[Tuple[NFA, bool]] = []
        for minimal_cover in all_min_covers:
            if self.store_progess:
                self.steps['Cover Enumeration'].append(minimal_cover)
            if self.verbose:
                print(f"\nFound a minimal cover, with size {len(minimal_cover)}:\n\n{'\n'.join([str(i) for i in minimal_cover])}\n")    
                print("Applying Intersection Rule")
            
            minimized_nfa = self.apply_intersection_rule(dfa, rsm, minimal_cover)
            is_legitimate = minimized_nfa.is_equivalent_to(self)
            
            if self.store_progess:
                self.steps['Intersection Rule Result'].append(minimized_nfa)
                self.steps['Legitimacy'].append(is_legitimate)
            if self.verbose:
                print(f"\nNFA from Intersection Rule:\n{minimized_nfa}\n\nThis NFA is {'not ' if not is_legitimate else ''}legitimate (i.e. recognizes same language as Original NFA!).\n")
            
            if is_legitimate:
                if miiini is None:
                    miiini = minimized_nfa
                elif len(minimized_nfa.States) < len(miiini.States):
                    miiini = minimized_nfa
            if miiini and not get_all_nfas:
                return miiini
            else:
                all_nfas.append((minimized_nfa, is_legitimate))

        if not get_all_nfas:
            return self
        else:
            return all_nfas
