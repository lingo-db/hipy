from typing import Set

import hipy.ir as ir


def enumerate_used_symbols(block: ir.Block, used_symbols: Set[str], module):
    def add_symbol(fn_name: str):
        if fn_name not in used_symbols:
            used_symbols.add(fn_name)
            enumerate_used_symbols(module.func(fn_name).body, used_symbols, module)

    for op in block.ops:
        match op:
            case ir.Call(name=fn_name):
                add_symbol(fn_name)
            case ir.FunctionRef(name=fn_name):
                add_symbol(fn_name)
        for nested_block in op.get_nested_blocks():
            enumerate_used_symbols(nested_block, used_symbols, module)


def run(module: ir.Module, root: str):
    used_symbols = set()
    used_symbols.add(root)
    root_fn = module.func(root)
    enumerate_used_symbols(root_fn.body, used_symbols, module)
    funcs_to_keep = [module.func(fn) for fn in used_symbols]
    module.block.ops = funcs_to_keep
