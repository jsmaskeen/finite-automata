from src.fa.nfa import NFA
from src.fa.dfa import DFA
from src.custom_typing.custom_types import STATE, StatesMap
from src.utils.helper import Helper
from typing import (
    List,
    Set,
    FrozenSet,
    Dict,
    DefaultDict,
    Any,
    Optional,
    Tuple,
    overload,
    Literal,
)
import collections
import itertools


class KamedaWeinerMinimize(NFA):
    """State minimization of an NFA using the Kameda-Weiner algorithm.

    Extends NFA to perform the Kameda-Weiner synthesis procedure, which
    finds a minimum-state NFA equivalent to the given NFA by analyzing
    the reduced automaton matrix (RAM), identifying prime grids, and
    applying the intersection rule over minimal covers.

    Inherits all standard NFA functionality and adds methods specific
    to the Kameda-Weiner minimization steps.
    """

    def __init__(self, nfa: NFA, store_progress: bool = False, verbose: bool = False):
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
        self.verbose: bool = verbose
        self.steps: Dict[str, Any] = {
            "Original NFA": nfa,
            "Reverse NFA": NFA(),
            "DFA": DFA(),
            "Dual of DFA": DFA(),
            "States Map Reduction": [],
            "Maximal Prime Grids": [],
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
            dfa_map (Dict[int, FrozenSet[STATE]]): Mapping from DFA state ID to its corresponding NFA-state subset.
            dfa_dual_map (Dict[int, FrozenSet[STATE]]): Mapping from dual-DFA state ID to its corresponding NFA-state subset.

        Returns:
            StatesMap: A dict containing:
                - `rows`: list of singleton frozensets for each DFA state
                - `cols`: list of singleton frozensets for each dual DFA state
                - `matrix`: 2D list where each cell is the intersection (FrozenSet) of the corresponding row and column, or None if empty
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
        counter: int = 1
        if self.verbose:
            print("\nReducing States Map matrix\n")
        while True:
            if self.store_progess:
                self.steps["States Map Reduction"].append(current_map)
            if self.verbose:
                print(f"\nStep {counter}:")
                Helper.print_states_map(current_map)
                counter += 1

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
        q = collections.deque([initial_grid])
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
            self.steps["Maximal Prime Grids"] = maxi_grids
        if self.verbose:
            print("Found Maximal Grids:\n" + '\n'.join([f"Rows: {set(i)}\nColumns: {set(j)}" for i, j in maxi_grids]) + '\n')
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
            self.steps["DFA"] = (dfa, dfa_map)

        if self.verbose:
            print(f"Input NFA for some L:\n{self}\n")
            print(f"DFA for L:\n{dfa}\nMapping:\n{dfa_map}\n")

        nfa_rev = self.reversal()
        dual_dfa, dfa_dual_map = nfa_rev.toDFA(True)

        if self.store_progess:
            self.steps["Dual of DFA"] = (dual_dfa, dfa_dual_map)
            self.steps["Reverse NFA"] = nfa_rev

        if self.verbose:
            print(f"NFA for rev(L):\n{nfa_rev}\n")
            print(f"DFA for rev(L):\n{dual_dfa}\nMapping:\n{dfa_dual_map}\n")

        states_map = self.construct_states_map(dfa_map, dfa_dual_map)

        if self.store_progess:
            self.steps["States Map Reduction"].append(states_map)

        if self.verbose:
            Helper.print_states_map(states_map)

        rsm = self.reduce_states_map(states_map)
        if self.verbose:
            print(
                "\nSuccessfully reduced the States Map matrix. You can see above the RSM.\n"
            )
        prime_grids = self.find_maximal_prime_grids(rsm)

        all_min_covers = self.find_all_minimal_covers(rsm, prime_grids)

        if not all_min_covers:
            if self.verbose:
                print(
                    "\nNo minimal cover found! Given NFA cannot be reduced using Kameda-Weiner\n"
                )
            return self
        miiini = None
        all_nfas: List[Tuple[NFA, bool]] = []
        for minimal_cover in all_min_covers:
            if self.store_progess:
                self.steps["Cover Enumeration"].append(minimal_cover)
            if self.verbose:
                joined = "\n".join(str(i) for i in minimal_cover)
                print(f"\nFound a minimal cover, with size {len(minimal_cover)}:\n\n{joined}\n")
                print("Applying Intersection Rule")

            minimized_nfa = self.apply_intersection_rule(dfa, rsm, minimal_cover)
            is_legitimate = minimized_nfa.is_equivalent_to(self)

            if self.store_progess:
                self.steps["Intersection Rule Result"].append(minimized_nfa)
                self.steps["Legitimacy"].append(is_legitimate)
            if self.verbose:
                print(
                    f"\nNFA from Intersection Rule:\n{minimized_nfa}\n\nThis NFA is {'not ' if not is_legitimate else ''}legitimate (i.e. recognizes same language as Original NFA!).\n"
                )

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
