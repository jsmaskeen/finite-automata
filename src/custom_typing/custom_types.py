from typing import Union, Optional, List, FrozenSet, TypedDict, DefaultDict, Set

STATE = Union[str, int]
"""A type alias for representing either a string or integer state identifier."""

class StatesMap(TypedDict):
    """Mapping of DFA state partitions used in the Kameda-Weiner minimization algorithm.

    In the context of the Kameda-Weiner algorithm, the states map matrix records how
    groups of DFA states (rows and columns) intersect during minimization. This structure
    supports tracking which combined state sets remain distinguishable or mergeable.
    """

    rows: List[FrozenSet[int]]
    """Sequence of frozen sets, each representing a block
    of DFA state indices corresponding to a row partition
    in the states map.
    """
    cols: List[FrozenSet[int]]
    """Sequence of frozen sets, each representing a block
    of DFA state indices corresponding to a column partition
    in the states map.
    """
    matrix: List[List[Optional[FrozenSet[STATE]]]]
    """A two-dimensional list with dimensions len(rows) x len(cols).
    Each cell holds either:
    - A frozen set of STATE elements indicating the intersection of the
    respective row's and column's state blocks during minimization.
    - None if no intersection (i.e., the blocks do not share any states).
    """


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
        transitions = [
            f"'{sym}': {{{', '.join(map(str, sorted(dests)))}}}"
            for sym, dests in self.items()
        ]
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
        lines: List[str] = []
        for src in sorted(self.keys()):
            transitions = self[src]
            trans_str = ", ".join(
                f"'{sym}': {{{', '.join(map(str, sorted(dests)))}}}"
                for sym, dests in transitions.items()
            )
            lines.append(f"{src} -> {{{trans_str}}}")
        return "\n".join(lines)
