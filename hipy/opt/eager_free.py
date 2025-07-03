import hipy.ir as ir


def insert_free_func(fn: ir.Function):
    free_ops = []
    used = set()
    for op in reversed(fn.body.ops):
        local_free_ops = []
        if not (isinstance(op, ir.Return) or isinstance(op, ir.Yield)):
            used_values = op.get_used_values()
            for v in used_values:
                if v not in used:
                    used.add(v)
                    local_free_ops.append(v)
        free_ops.append(local_free_ops)
    new_block = ir.Block()
    for op, free in zip(fn.body.ops, reversed(free_ops)):
        new_block.ops.append(op)
        for free in free:
            ir.Free(new_block, free)
    fn.body = new_block


def run(module: ir.Module):
    for fn in module.block.ops:
        insert_free_func(fn)
