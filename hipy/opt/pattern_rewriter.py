import sys
from abc import ABC, abstractmethod
from collections import deque

import hipy.ir as ir
from typing import Dict, List, Set


class PatternRewriter:
    def __init__(self, patterns: List["RewritePattern"], module: ir.Module):
        self.uses: Dict[ir.SSAValue, Set[ir.Operation]] = {}
        self.patterns = patterns
        self.module = module
        self.current_op = None
        self.current_block = None
        def rec_uses(op):
            for v in op.get_used_values():
                if v not in self.uses:
                    self.uses[v] = set()
                self.uses[v].add(op)
            for b in op.get_nested_blocks():
                for op in b.ops:
                    rec_uses(op)
        for fn in module.block.ops:
            for op in fn.body.ops:
                rec_uses(op)


        self.d = deque()
        for fn in module.block.ops:
            for op in fn.body.ops:
                self.d.append((op, fn.body))

    def rewrite(self):
        while len(self.d) > 0:
            op, block = self.d.popleft()
            self.current_block = block
            self.current_op = op
            any_matched=False
            for pattern in self.patterns:
                if pattern.rewrite(op, self):
                    any_matched=True
                    break

            if not any_matched:
                for b in op.get_nested_blocks():
                    for op in b.ops:
                        self.d.append((op, b))

    def remove(self, op: ir.Operation):
        self.current_block.ops.remove(op)
        if (op, self.current_block) in self.d:
            self.d.remove((op, self.current_block))
        for v in op.get_used_values():
            if v in self.uses and op in self.uses[v]:
                self.uses[v].remove(op)


    def maybe_remove(self, op: ir.Operation):
        for v in op.get_produced_values():
            if v in self.uses and len(self.uses[v]) > 0:
                return
        if op.has_side_effects():
            return
        self.remove(op)

    def replace_with_value(self, op: ir.Operation, value: ir.SSAValue):
        assert len(op.get_produced_values()) == 1
        if op.get_produced_values()[0] in self.uses:
            for use in self.uses[op.get_produced_values()[0]]:
                use.replace_uses(op.get_produced_values()[0], value)
                self.uses.setdefault(value, set()).add(use)
        #else:
        #    print("No uses for", op.get_produced_values()[0],file=sys.stderr)

        self.remove(op)

    def update_uses(self, op: ir.Operation):
        for v in op.get_used_values():
            if v not in self.uses:
                self.uses[v] = set()
            self.uses[v].add(op)

    def replace_with(self, op: ir.Operation, cls, *args, **kwargs):
        new_op = cls(self.before_current(), *args, **kwargs)

        assert len(op.get_produced_values()) == len(new_op.get_produced_values())
        for old, new in zip(op.get_produced_values(), new_op.get_produced_values()):
            assert old.type == new.type
            if new not in self.uses:
                self.uses[new] = set()
            if old in self.uses:
                for use in self.uses[old]:
                    use.replace_uses(old, new)
                    self.uses[new].add(use)
        for v in new_op.get_used_values():
            if v not in self.uses:
                self.uses[v] = set()
            self.uses[v].add(new_op)
        for v in op.get_used_values():
            if v in self.uses and op in self.uses[v]:
                self.uses[v].remove(op)
            if v in self.uses and len(self.uses[v]) == 0:
                del self.uses[v]
                self.maybe_remove(v.producer)

        self.d.appendleft((new_op, self.current_block))
        self.remove(op)
        return new_op

    def before_current(self):
        rewriter = self

        class BeforeCurrent:
            def append(self, op: ir.Operation):
                rewriter.current_block.ops.insert(rewriter.current_block.ops.index(rewriter.current_op), op)

            def __getattr__(self, name):
                if name == "ops":
                    return self
                return getattr(self, name)

        return BeforeCurrent()
    def before(self,beforeop:ir.Operation):
        rewriter = self

        class BeforeCurrent:
            def append(self, op: ir.Operation):
                rewriter.current_block.ops.insert(rewriter.current_block.ops.index(beforeop), op)

            def __getattr__(self, name):
                if name == "ops":
                    return self
                return getattr(self, name)

        return BeforeCurrent()

    def create(self, cls, *args, **kwargs):
        new_op = cls(self.before_current(), *args, **kwargs)
        for v in new_op.get_used_values():
            if v not in self.uses:
                self.uses[v] = set()
            self.uses[v].add(new_op)
        self.d.append((new_op, self.current_block))
        return new_op


class RewritePattern(ABC):
    @abstractmethod
    def rewrite(self, op: ir.Operation, rewriter: PatternRewriter):
        pass
