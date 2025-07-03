import hipy.ir as ir


def eliminate_dead_code_(block: ir.Block):
    for op in block.ops:
        for nested_block in op.get_nested_blocks():
            eliminate_dead_code_(nested_block)
    used_values = set()
    dead_ops = set()
    for op in reversed(block.ops):
        if isinstance(op, ir.Return) or isinstance(op, ir.Yield):
            used_values.update(op.get_used_values())
        else:
            if op.has_side_effects():
                used_values.update(op.get_used_values())
            else:
                if any([v in used_values for v in op.get_produced_values()]):
                    used_values.update(op.get_used_values())
                else:
                    dead_ops.add(op)
    for op in dead_ops:
        block.ops.remove(op)


def run(module: ir.Module):
    for fn in module.block.ops:
        eliminate_dead_code_(fn.body)
