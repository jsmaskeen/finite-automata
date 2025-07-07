import collections

class DFA:
    def __init__(self):
        self.States = set()
        self.Sigma = set()
        self.Initial = None
        self.Final = set()
        self.delta = {}

    def addState(self, state):
        self.States.add(state)

    def setSigma(self, sigma_set):
        self.Sigma = sigma_set

    def setInitial(self, state):
        self.Initial = state
        self.addState(state)

    def addFinal(self, state):
        self.Final.add(state)

    def addTransition(self, src, char, dest):
        self.delta[(src, char)] = dest
        self.addState(src)
        self.addState(dest)

    def toNFA(self):
        nfa = NFA()
        nfa.setSigma(self.Sigma)
        for s in self.States:
            nfa.addState(s)
        nfa.setInitial(self.Initial)
        for f in self.Final:
            nfa.addFinal(f)
        for (src, char), dest in self.delta.items():
            nfa.addTransition(src, char, dest)
        return nfa

    def reversal(self):
        rev_nfa = NFA()
        rev_nfa.setSigma(self.Sigma)
        new_initial_state_set = self.Final
        rev_nfa.setInitial(new_initial_state_set)

        rev_nfa.addFinal(self.Initial)

        for (src, char), dest in self.delta.items():
            rev_nfa.addTransition(dest, char, src)

        for s in self.States:
            rev_nfa.addState(s)

        return rev_nfa

    def minimal(self):
        states = sorted(list(self.States))
        distinguishable = set()
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                p, q = states[i], states[j]
                if (p in self.Final and q not in self.Final) or (
                    p not in self.Final and q in self.Final
                ):
                    distinguishable.add(tuple(sorted((p, q))))

        while True:
            newly_marked = set()
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    p, q = states[i], states[j]
                    if tuple(sorted((p, q))) in distinguishable:
                        continue

                    for char in self.Sigma:
                        next_p = self.delta.get((p, char))
                        next_q = self.delta.get((q, char))

                        if (
                            next_p is not None
                            and next_q is not None
                            and next_p != next_q
                        ):
                            if tuple(sorted((next_p, next_q))) in distinguishable:
                                newly_marked.add(tuple(sorted((p, q))))
                                break

            if not newly_marked:
                break
            distinguishable.update(newly_marked)

        all_pairs = {
            tuple(sorted((states[i], states[j])))
            for i in range(len(states))
            for j in range(i + 1, len(states))
        }
        indistinguishable_pairs = all_pairs - distinguishable

        parent = {s: s for s in self.States}

        def find(s):
            if parent[s] == s:
                return s
            parent[s] = find(parent[s])
            return parent[s]

        def union(s1, s2):
            root1 = find(s1)
            root2 = find(s2)
            if root1 != root2:
                parent[root2] = root1

        for p, q in indistinguishable_pairs:
            union(p, q)

        partitions = collections.defaultdict(list)
        for s in self.States:
            partitions[find(s)].append(s)

        min_dfa = DFA()
        min_dfa.setSigma(self.Sigma)

        partition_map = {
            frozenset(v): f"q{i}" for i, v in enumerate(partitions.values())
        }
        state_to_partition_name = {
            s: partition_map[frozenset(partitions[find(s)])] for s in self.States
        }

        for p_name in partition_map.values():
            min_dfa.addState(p_name)

        min_dfa.setInitial(state_to_partition_name[self.Initial])
        for f in self.Final:
            min_dfa.addFinal(state_to_partition_name[f])

        for p_frozenset, p_name in partition_map.items():
            representative = next(iter(p_frozenset))
            for char in self.Sigma:
                if (representative, char) in self.delta:
                    next_state = self.delta[(representative, char)]
                    next_partition_name = state_to_partition_name[next_state]
                    min_dfa.addTransition(p_name, char, next_partition_name)

        return min_dfa
    
    def is_isomorphic_to(self, other_dfa):
        if self.Sigma != other_dfa.Sigma or \
           len(self.States) != len(other_dfa.States) or \
           len(self.Final) != len(other_dfa.Final):
            return False

        # BFS time !!!
        queue = collections.deque([(self.Initial, other_dfa.Initial)])
        visited_map = {self.Initial: other_dfa.Initial}

        while queue:
            q1, q2 = queue.popleft()

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


class NFA:
    def __init__(self):
        self.States = set()
        self.Sigma = set()
        self.Initial = set()
        self.Final = set()
        self.delta = collections.defaultdict(lambda: collections.defaultdict(set))

    def addState(self, state):
        self.States.add(state)

    def setSigma(self, sigma_set):
        self.Sigma = sigma_set

    def setInitial(self, states):
        if not isinstance(states, (list, set, frozenset)):
            states = {states}
        self.Initial = set(states)
        self.States.update(self.Initial)

    def initialSet(self):
        return self.Initial

    def addFinal(self, state):
        self.Final.add(state)
        self.addState(state)

    def addTransition(self, src, char, dest):
        self.delta[src][char].add(dest)
        self.addState(src)
        self.addState(dest)

    def reversal(self):
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

    def toDFA(self):

        dfa = DFA()
        dfa.setSigma(self.Sigma)

        if not self.Initial:
            return dfa

        initial_dfa_state = frozenset(self.Initial)

        unprocessed_states = collections.deque([initial_dfa_state])
        dfa_states = {initial_dfa_state}

        state_map = {initial_dfa_state: "S0"}

        dfa.setInitial(state_map[initial_dfa_state])

        while unprocessed_states:
            current_nfa_states = unprocessed_states.popleft()
            current_dfa_name = state_map[current_nfa_states]

            if not self.Final.isdisjoint(current_nfa_states):
                dfa.addFinal(current_dfa_name)

            for char in self.Sigma:
                next_nfa_states = set()
                for nfa_state in current_nfa_states:
                    next_nfa_states.update(
                        self.delta.get(nfa_state, {}).get(char, set())
                    )

                if not next_nfa_states:
                    continue

                next_nfa_states_fs = frozenset(next_nfa_states)

                if next_nfa_states_fs not in dfa_states:
                    dfa_states.add(next_nfa_states_fs)
                    unprocessed_states.append(next_nfa_states_fs)
                    state_map[next_nfa_states_fs] = f"S{len(state_map)}"

                next_dfa_name = state_map[next_nfa_states_fs]
                dfa.addTransition(current_dfa_name, char, next_dfa_name)

        for name in state_map.values():
            dfa.addState(name)

        return dfa

    def is_equivalent_to(self, other_nfa):
        min_dfa1 = self.toDFA().minimal()
        min_dfa2 = other_nfa.toDFA().minimal()
        return min_dfa1.is_isomorphic_to(min_dfa2)