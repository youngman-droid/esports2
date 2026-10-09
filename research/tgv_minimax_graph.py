"""Vectorized min/max DAG evaluation, retaining active terminal derivatives."""
import numpy as np


class MinimaxGraph:
    def __init__(self, terminal_count, nodes, roots):
        self.terminal_count = terminal_count
        self.size = terminal_count + len(nodes)
        self.roots = np.asarray(roots)
        depths = np.zeros(self.size, dtype=int)
        buckets = {}
        for i, (maximum, children) in enumerate(nodes, terminal_count):
            children = np.asarray(children, dtype=int)
            assert len(children) and np.all(children < i)
            depth = 1 + max(depths[children])
            depths[i] = depth
            buckets.setdefault((depth, maximum), []).append((i, children))
        self.layers = []
        for (depth, maximum), rows in sorted(buckets.items()):
            indices = np.array([i for i, _ in rows])
            lengths = np.array([len(c) for _, c in rows])
            starts = np.r_[0, np.cumsum(lengths)[:-1]]
            children = np.concatenate([c for _, c in rows])
            self.layers.append((maximum, indices, lengths, starts, children))

    def evaluate(self, terminals):
        values = np.empty(self.size)
        active = np.empty(self.size, dtype=int)
        values[:self.terminal_count] = terminals
        active[:self.terminal_count] = np.arange(self.terminal_count)
        for maximum, indices, lengths, starts, children in self.layers:
            cv = values[children]
            result = (np.maximum if maximum else np.minimum).reduceat(cv, starts)
            values[indices] = result
            eligible = cv == np.repeat(result, lengths)
            active[indices] = np.minimum.reduceat(
                np.where(eligible, active[children], self.terminal_count), starts)
        return values[self.roots], active[self.roots]

    def smooth(self, terminals, terminal_jacobian, temperature):
        """Log-sum-exp continuation; it approaches exact extrema as tau -> 0.

        This is an optimization device, not a probabilistic draft policy.
        """
        assert temperature > 0
        values = np.empty(self.size)
        jacobian = np.empty((self.size, terminal_jacobian.shape[1]))
        values[:self.terminal_count] = terminals
        jacobian[:self.terminal_count] = terminal_jacobian
        for maximum, indices, lengths, starts, children in self.layers:
            sign = 1 if maximum else -1
            cv = sign * values[children]
            offset = np.maximum.reduceat(cv, starts)
            exp = np.exp((cv - np.repeat(offset, lengths)) / temperature)
            total = np.add.reduceat(exp, starts)
            values[indices] = sign * (offset + temperature * np.log(total))
            weight = exp / np.repeat(total, lengths)
            jacobian[indices] = np.add.reduceat(
                jacobian[children] * weight[:, None], starts, axis=0)
        return values[self.roots], jacobian[self.roots]
