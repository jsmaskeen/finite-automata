from typing import Any, Optional, FrozenSet, Union, List
from src.custom_typing.custom_types import STATE, StatesMap


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
    def print_states_map(sm: StatesMap):
        """
        Helper function to pretty print a StatesMap object.

        Args:
            sm (StatesMap):
                The StatesMap to print.
        """

        def fmt(entry: Optional[FrozenSet[Union[int, STATE]]]) -> str:
            if entry is None:
                return "\u03c6"
            return "{" + ", ".join(sorted(map(str, entry))) + "}"

        rows = [fmt(r) for r in sm["rows"]]
        cols = [fmt(c) for c in sm["cols"]]
        mat = sm["matrix"]
        pad: int = 4
        table = [
            [("\u03c6" if cell is None else fmt(cell)) for cell in row] for row in mat
        ]

        w0 = max(len(s) for s in rows + [""]) + pad

        col_widths: List[int] = []
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
