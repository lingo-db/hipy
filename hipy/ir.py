import ast
from typing import List, Any, Tuple, Dict
import abc


class VoidType:
    def __str__(self):
        return 'void'

    def serialize(self):
        return {"kind": "type",
                "name": "void"
                }

    def mangle(self):
        return "void"

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()



class PyObjType:
    def __str__(self):
        return 'py_object'

    def serialize(self):
        return {"kind": "type",
                "name": "py_object"
                }

    def mangle(self):
        return "pyobj"

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return self


class BoolType:
    def __str__(self):
        return 'bool'

    def serialize(self):
        return {"kind": "type",
                "name": "bool"
                }

    def mangle(self):
        return "bool"

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return self


class IntegerType:
    def __init__(self, width):
        self.width = width

    def __str__(self):
        return f'i{self.width}'

    def serialize(self):
        return {"kind": "type",
                "name": "int",
                "width": self.width
                }

    def mangle(self):
        return str(self)

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return IntegerType(None)


class FloatType:
    def __init__(self, width: int):
        self.width = width

    def __str__(self):
        return f'f{self.width}'

    def serialize(self):
        return {"kind": "type",
                "name": "float",
                "width": self.width
                }

    def mangle(self):
        return str(self)

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return FloatType(None)


class StringType:
    def __str__(self):
        return 'str'

    def serialize(self):
        return {"kind": "type",
                "name": "string"
                }

    def mangle(self):
        return str(self)

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return self


class IntType:
    def __str__(self):
        return 'int'

    def serialize(self):
        return {"kind": "type",
                "name": "pyint"
                }

    def mangle(self):
        return str(self)

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return self


class DictType:
    def __init__(self, key_type, val_type):
        self.key_type = key_type
        self.val_type = val_type

    def __str__(self):
        return f"dict[{self.key_type},{self.val_type}]"

    def serialize(self):
        return {"kind": "type",
                "name": "dict",
                "key_type": self.key_type.serialize(),
                "val_type": self.val_type.serialize(),
                }

    def mangle(self):
        return f"dict_{self.key_type.mangle()}_{self.val_type.mangle()}"

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()


class RecordType:
    def __init__(self, members: List[Tuple[str, Any]]):
        self.members = members
    def member_type(self, col):
        for m, t in self.members:
            if m == col:
                return t
        return None
    def __str__(self):
        return f'record[{", ".join(map(lambda m: f'{m[0]}: {m[1]}', self.members))}]'

    def mangle(self):
        return f"record_{"_".join([f"{m}_{t.mangle()}" for m, t in self.members])}"

    def serialize(self):
        return {"kind": "type",
                "name": "record",
                "members": {m: t.serialize() for m, t in self.members}
                }

    def get_generic(self):
        return RecordType([])

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        if not isinstance(other, RecordType):
            return False
        return self.members == other.members


class ArrayType:
    def __init__(self, element_type, shape=None):
        self.element_type = element_type
        self.shape = shape

    def shape_str(self):
        res = "x".join(["?" if s is None else str(s) for s in self.shape])
        return res

    def __str__(self):
        return f'array[{self.shape_str() if self.shape is not None else ""}x{self.element_type}]'

    def serialize(self):
        return {"kind": "type",
                "name": "array",
                "element_type": self.element_type.serialize(),
                }


class ColumnType:
    def __init__(self, element_type):
        self.element_type = element_type

    def __str__(self):
        return f'column[{self.element_type}]'

    def mangle(self):
        return f"column_{self.element_type.mangle()}" if self.element_type is not None else "column"  # todo incorporate shape

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return ColumnType(None)

    def serialize(self):
        return {"kind": "type",
                "name": "column",
                "element_type": self.element_type.serialize(),
                }


class ListType:
    def __init__(self, element_type):
        self.element_type = element_type

    def __str__(self):
        return f'list[{self.element_type}]'

    def mangle(self):
        return f"list_{self.element_type.mangle()}" if self.element_type is not None else "list"  # todo incorporate shape

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def get_generic(self):
        return ListType(None)

    def serialize(self):
        return {"kind": "type",
                "name": "list",
                "element_type": self.element_type.serialize(),
                }


class TableType:
    def __init__(self, members: List[Tuple[str, Any]]):
        self.members = members

    def __str__(self):
        return f'table[{",".join(map(lambda m: f'{m[0]}: {m[1]}', self.members))}]'

    def col_type(self, col):
        for m, t in self.members:
            if m == col:
                return t
        return None

    def get_generic(self):
        return TableType(None)

    def mangle(self):
        return "table" if self.members is None else f"table_{"_".join([f"{m}_{t.mangle()}" for m, t in self.members])}"

    def __hash__(self):
        return hash(self.mangle())

    def __eq__(self, other):
        return self.mangle() == other.mangle()

    def serialize(self):
        return {"kind": "type",
                "name": "table",
                "members": {m: t.serialize() for m, t in self.members}
                }


type DBPyType = IntegerType | FloatType

void = VoidType()
bool = BoolType()
i8 = IntegerType(8)
i16 = IntegerType(16)
i32 = IntegerType(32)
i64 = IntegerType(64)
int = IntType()
f32 = FloatType(32)
f64 = FloatType(64)
string = StringType()
pyobj = PyObjType()

ssa_value_ctr = 0


class SSAValue:
    def __init__(self, type: DBPyType, producer):
        global ssa_value_ctr
        self.id = ssa_value_ctr
        ssa_value_ctr += 1
        self.type = type
        self.producer = producer

    def __str__(self):
        return f'val_{self.id} : {self.type}'

    def serialize(self):
        return {
            "kind": "value",
            "id": self.id,
            "type": self.type.serialize()
        }

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id


block_ctr = 0


class Block:
    def __init__(self):
        global block_ctr
        self.ops: List[Operation] = []
        self.id = block_ctr
        block_ctr += 1

    def __str__(self):
        indent = '\t'
        lines = []
        for op in self.ops:
            lines.extend(str(op).split("\n"))
        return f'{indent}{f"\n{indent}".join(lines)}'

    def serialize(self):
        return list(map(lambda op: op.serialize(), self.ops))


class Module:
    def __init__(self):
        self.block = Block()
        self.imports = {}
        self.py_functions = {}

    def funcs(self):
        return self.block.ops

    def func(self, name):
        for f in self.funcs():
            if f.name == name:
                return f
        return None

    def __str__(self):
        return f"""module{{
[[IMPORTS]]
{self.imports}
[[PYTHON]]
{self.py_functions}
[[FUNCTIONS]]
{self.block}
}}"""

    def serialize(self):
        return {
            "kind": "module",
            "imports": self.imports,
            "py_functions": self.py_functions,
            "functions": self.block.serialize()
        }

    def merge(self, other: 'Module'):
        self.block.ops.extend(other.block.ops)
        self.imports.update(other.imports)
        self.py_functions.update(other.py_functions)


class Function:
    def __init__(self, module: Module, name: str, arg_types: List[DBPyType], res_type: DBPyType):
        module.block.ops.append(self)
        self.name = name
        self.arg_types = arg_types
        self.args = list(map(lambda argt: SSAValue(argt, self), arg_types))
        self.res_type = res_type
        self.body = Block()

    def __str__(self):
        return f"""func {self.name}({",".join(map(lambda arg: str(arg), self.args))}) -> {self.res_type} {{
{self.body}
}}"""

    def serialize(self):
        return {
            "kind": "function",
            "name": self.name,
            "args": list(map(lambda arg: arg.serialize(), self.args)),
            "res_type": self.res_type.serialize(),
            "body": self.body.serialize()
        }


class FunctionRefType:
    def __init__(self, function: Function, closure_type):
        self.arg_types = function.arg_types
        self.res_type = function.res_type
        self.closure_type = closure_type

    def __str__(self):
        return f'function_ref({",".join(map(lambda t: str(t), self.arg_types))})->{self.res_type}'

    def serialize(self):
        return {"kind": "type",
                "name": "function_ref",
                "res_type": self.res_type.serialize()
                }

    def mangle(self):
        return str(self).replace("(", "_").replace(")", "_").replace("->", "_")


lambdas = 0



class Operation(abc.ABC):
    @abc.abstractmethod
    def get_used_values(self):
        pass

    @abc.abstractmethod
    def get_produced_values(self):
        pass

    @abc.abstractmethod
    def has_side_effects(self):
        pass

    @abc.abstractmethod
    def replace_uses(self, old: SSAValue, new: SSAValue):
        pass

    def get_nested_blocks(self):
        return []

    @abc.abstractmethod
    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        pass


def replace_usage_in_list(l: List[SSAValue], old: SSAValue, new: SSAValue):
    for i, arg in enumerate(l):
        if arg == old:
            l[i] = new


def replace_usage_in_dict_values(d: Dict[Any, SSAValue], old: SSAValue, new: SSAValue):
    for k, v in d.items():
        if v == old:
            d[k] = new


class CallBuiltin(Operation):
    def __init__(self, block: Block, name: str, args: List[SSAValue], ret_type: DBPyType, side_effects=True, attributes=None):
        block.ops.append(self)
        self.name = name
        self.args = args
        self.ret_type = ret_type
        self.result = SSAValue(ret_type, self)
        self.side_effects = side_effects
        self.attributes=attributes

    def __str__(self):
        return f'{self.result} = builtin {"[noeffect]" if not self.side_effects else ""} {self.name}({", ".join(map(lambda x: str(x), self.args))}) {self.attributes if self.attributes is not None else ""}'

    def serialize(self):
        return {
            "kind": "CallBuiltin",
            "name": self.name,
            "args": list(map(lambda arg: arg.serialize(), self.args)),
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return self.args

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return self.side_effects

    def replace_uses(self, old: SSAValue, new: SSAValue):
        return replace_usage_in_list(self.args, old, new)

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = CallBuiltin(block, self.name, [mapping[a] if a in mapping else a for a in self.args], self.ret_type, self.side_effects)
        mapping[self.result] = cloned.result
        return cloned


class FunctionRef(Operation):
    def __init__(self, block: Block, func: Function, closure=None):
        block.ops.append(self)
        self.name = func.name
        self.return_type = FunctionRefType(func, closure.type if closure is not None else None)
        self.closure = closure
        self.result = SSAValue(self.return_type, self)
        self.func=func

    def __str__(self):
        return f'{self.result} = function_ref {self.name}' + (f'[{self.closure}]' if self.closure is not None else '')

    def serialize(self):
        d = {
            "kind": "FunctionRef",
            "name": self.name,
            "result": self.result.serialize(),

        }
        if self.closure is not None:
            d["closure"] = self.closure.serialize()
        return d

    def get_used_values(self):
        return [self.closure] if self.closure is not None else []

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return False

    def replace_uses(self, old: SSAValue, new: SSAValue):
        if self.closure is not None and self.closure == old:
            self.closure = new

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = FunctionRef(block, self.func, mapping[self.closure] if self.closure is not None else None)
        mapping[self.result] = cloned.result
        return cloned


class MakeRecord(Operation):
    def __init__(self, block: Block, res_type: RecordType, values: dict[str, SSAValue]):
        for v in values.values():
            assert isinstance(v,SSAValue)
        block.ops.append(self)
        self.values = values
        self.res_type = res_type
        self.result = SSAValue(res_type, self)

    def __str__(self):
        return f'{self.result} = make_record {", ".join(map(lambda i: f"{i[0]}={str(i[1])}", self.values.items()))}'

    def serialize(self):
        return {
            "kind": "MakeRecord",
            "values": {k: v.serialize() for k, v in self.values.items()},
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return [v for _, v in self.values.items()]

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return False

    def replace_uses(self, old: SSAValue, new: SSAValue):
        return replace_usage_in_dict_values(self.values, old, new)

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = MakeRecord(block, self.res_type, {k: mapping[v] if v in mapping else v for k, v in self.values.items()})
        mapping[self.result] = cloned.result
        return cloned


class RecordGet(Operation):
    def __init__(self, block: Block, res_type, record: SSAValue, member: str):
        assert isinstance(record.type, RecordType)
        block.ops.append(self)
        self.member = member
        self.record = record
        self.res_type = res_type
        self.result = SSAValue(res_type, self)

    def __str__(self):
        return f'{self.result} = record_get {self.record}[{self.member}]'

    def serialize(self):
        return {
            "kind": "RecordGet",
            "record": self.record.serialize(),
            "member": self.member,
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return [self.record]

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return False

    def replace_uses(self, old: SSAValue, new: SSAValue):
        if self.record == old:
            self.record = new

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = RecordGet(block, self.res_type, mapping[self.record] if self.record in mapping else self.record, self.member)
        mapping[self.result] = cloned.result
        return cloned


class PyImport(Operation):
    def __init__(self, block: Block, name: str):
        block.ops.append(self)
        self.name = name
        self.result = SSAValue(pyobj, self)

    def __str__(self):
        return f'{self.result} = py_import {self.name}'

    def serialize(self):
        return {
            "kind": "PyImport",
            "name": self.name,
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return []

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return False  # todo: check

    def replace_uses(self, old: SSAValue, new: SSAValue):
        pass

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = PyImport(block, self.name)
        mapping[self.result] = cloned.result
        return cloned



class PythonCall(Operation):
    def __init__(self, block: Block, callable: SSAValue, args: List[SSAValue], kw_args: List[Tuple[str, SSAValue]]):
        block.ops.append(self)
        self.callable = callable
        self.args = args
        self.kw_args = kw_args
        self.result = SSAValue(pyobj, self)

    def __str__(self):
        return f'{self.result} = py_call ({self.callable})({", ".join(map(lambda x: str(x), self.args)) + "," if len(self.args) > 0 else ""}{",".join(map(lambda x: f"{x[0]}={x[1]}", self.kw_args))})'

    def serialize(self):
        return {
            "kind": "PyCall2",
            "callable": self.callable.serialize(),
            "args": list(map(lambda arg: arg.serialize(), self.args)),
            "kw_args": [{"name": k, "value": v.serialize()} for k, v in self.kw_args],
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return [self.callable]+self.args + [v for _, v in self.kw_args]

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        replace_usage_in_list(self.args, old, new)
        for i, (k, v) in enumerate(self.kw_args):
            if v == old:
                self.kw_args[i] = (k, new)

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = PythonCall(block, mapping[self.callable], [mapping[a] for a in self.args],
                            [(k, mapping[v]) for k, v in self.kw_args])
        mapping[self.result] = cloned.result
        return cloned


class PyGetAttr(Operation):
    def __init__(self, block: Block, name: str, on: SSAValue):
        block.ops.append(self)
        self.name = name
        self.result = SSAValue(pyobj, self)
        self.on = on

    def __str__(self):
        return f'{self.result} = py_get_attr ({self.on}).{self.name}'

    def serialize(self):
        return {
            "kind": "PyGetAttr",
            "name": self.name,
            "on": self.on.serialize(),
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return [self.on]

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        if self.on == old:
            self.on = new

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = PyGetAttr(block, self.name, mapping[self.on])
        mapping[self.result] = cloned.result
        return cloned


class PySetAttr(Operation):
    def __init__(self, block: Block, name: str, on: SSAValue, value: SSAValue):
        block.ops.append(self)
        self.name = name
        self.result = SSAValue(void, self)
        self.value = value
        self.on = on

    def __str__(self):
        return f'{self.result} = py_set_attr ({self.on}).{self.name} = {self.value}'

    def serialize(self):
        return {
            "kind": "PySetAttr",
            "name": self.name,
            "on": self.on.serialize(),
            "value": self.value.serialize(),
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return [self.on, self.value]

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        if self.on == old:
            self.on = new
        if self.value == old:
            self.value = new

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = PySetAttr(block, self.name, mapping[self.on], mapping[self.value])
        mapping[self.result] = cloned.result
        return cloned


class Call(Operation):
    def __init__(self, block: Block, name: str, args: List[SSAValue], ret_type: DBPyType):
        block.ops.append(self)
        self.name = name
        self.args = args
        self.ret_type = ret_type
        self.result = SSAValue(ret_type, self)

    def __str__(self):
        return f'{self.result} = call {self.name}({", ".join(map(lambda x: str(x), self.args))})'

    def serialize(self):
        return {
            "kind": "Call",
            "name": self.name,
            "args": list(map(lambda arg: arg.serialize(), self.args)),
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return self.args

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        replace_usage_in_list(self.args, old, new)

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = Call(block, self.name, [mapping[a] if a in mapping else a for a in self.args], self.ret_type)
        mapping[self.result] = cloned.result
        return cloned

class CallIndirect(Operation):
    def __init__(self, block: Block, fnref : SSAValue, args: List[SSAValue], ret_type: DBPyType):
        block.ops.append(self)
        self.fnref = fnref
        self.args = args
        self.ret_type = ret_type
        self.result = SSAValue(ret_type, self)

    def __str__(self):
        return f'{self.result} = call {self.fnref}({", ".join(map(lambda x: str(x), self.args))})'

    def serialize(self):
        return {
            "kind": "Call",
            "fnref": self.fnref,
            "args": list(map(lambda arg: arg.serialize(), self.args)),
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return self.args+[self.fnref]

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        replace_usage_in_list(self.args, old, new)
        if self.fnref == old:
            self.fnref = new

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = CallIndirect(block, mapping[self.fnref], [mapping[a] if a in mapping else a for a in self.args], self.ret_type)
        mapping[self.result] = cloned.result
        return cloned

class Constant(Operation):
    def __init__(self, block: Block, v, t: DBPyType):
        block.ops.append(self)
        self.result = SSAValue(t, self)
        self.v = v

    def __str__(self):
        return f'{self.result} =  const {repr(self.v)}'

    def serialize(self):
        return {
            "kind": "Constant",
            "value": self.v,
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return []

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return False

    def replace_uses(self, old: SSAValue, new: SSAValue):
        pass

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = Constant(block, self.v, self.result.type)
        mapping[self.result] = cloned.result
        return cloned


class Undef(Operation):
    def __init__(self, block: Block, t: DBPyType):
        block.ops.append(self)
        self.result = SSAValue(t, self)

    def __str__(self):
        return f'{self.result} =  undef'

    def serialize(self):
        return {
            "kind": "Undef",
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return []

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return False

    def replace_uses(self, old: SSAValue, new: SSAValue):
        pass

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = Undef(block, self.result.type)
        mapping[self.result] = cloned.result
        return cloned


class Free:
    def __init__(self, block: Block, v: SSAValue):
        block.ops.append(self)
        self.v = v
        self.result = SSAValue(void, self)

    def __str__(self):
        return f'{self.result} =  free {self.v}'

    def serialize(self):
        return {
            "kind": "NoOp",
            "result": self.result.serialize()
        }

    def get_used_values(self):
        return [self.v]

    def get_produced_values(self):
        return [self.result]

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        if self.v == old:
            self.v = new

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = Free(block, mapping[self.v])
        mapping[self.result] = cloned.result
        return cloned


class Return(Operation):
    def __init__(self, block: Block, values: List[SSAValue]):
        if len(block.ops) > 0 and isinstance(block.ops[-1], Return):
            raise Exception("Can't have multiple returns in a block")
        block.ops.append(self)
        self.values = values

    def __str__(self):
        return f'return {",".join(map(lambda v: str(v), self.values))}'

    def serialize(self):
        return {
            "kind": "Return",
            "values": list(map(lambda arg: arg.serialize(), self.values)),
        }

    def get_used_values(self):
        return self.values

    def get_produced_values(self):
        return []

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        replace_usage_in_list(self.values, old, new)
    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        assert False


class Yield(Operation):
    def __init__(self, block: Block, values: List[SSAValue]):
        block.ops.append(self)
        self.values = values

    def __str__(self):
        return f'yield {",".join(map(lambda v: str(v), self.values))}'

    def serialize(self):
        return {
            "kind": "Yield",
            "values": list(map(lambda arg: arg.serialize(), self.values)),
        }

    def get_used_values(self):
        return self.values

    def get_produced_values(self):
        return []

    def has_side_effects(self):
        return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        replace_usage_in_list(self.values, old, new)

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        return Yield(block, [mapping[v] if v in mapping else v for v in self.values])


def flatten(l):
    return [item for sublist in l for item in sublist]


class IfElse(Operation):
    def __init__(self, block: Block, cond: SSAValue, return_types: List[DBPyType]):
        block.ops.append(self)
        self.cond = cond
        self.ifBody = Block()
        self.elseBody = Block()
        self.results = list(map(lambda t: SSAValue(t, self), return_types))
        self.return_types = return_types

    def __str__(self):
        return f"""{", ".join(map(lambda v: str(v), self.results))}{"" if len(self.results) == 0 else " = "}if {self.cond} {"" if len(self.results) == 0 else " -> "}{", ".join(map(lambda v: str(v), self.return_types))} {{
{self.ifBody}
}} else {{
{self.elseBody}
}}"""

    def serialize(self):
        return {
            "kind": "IfElse",
            "cond": self.cond.serialize(),
            "results": list(map(lambda arg: arg.serialize(), self.results)),
            "ifBody": self.ifBody.serialize(),
            "elseBody": self.elseBody.serialize(),
        }

    def get_used_values(self):
        return [self.cond] + flatten([op.get_used_values() for op in self.ifBody.ops]) + flatten(
            [op.get_used_values() for op in
             self.elseBody.ops])

    def get_produced_values(self):
        return self.results

    def has_side_effects(self):
        for op in self.ifBody.ops:
            if op.has_side_effects():
                return True
        for op in self.elseBody.ops:
            if op.has_side_effects():
                return True

    def replace_uses(self, old: SSAValue, new: SSAValue):
        if self.cond == old:
            self.cond = new
        for op in self.ifBody.ops:
            op.replace_uses(old, new)
        for op in self.elseBody.ops:
            op.replace_uses(old, new)

    def get_nested_blocks(self):
        return [self.ifBody, self.elseBody]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        cloned = IfElse(block, mapping[self.cond], self.return_types)
        mapping.update({v: cloned_v for v, cloned_v in zip(self.results, cloned.results)})
        cloned.ifBody = Block()
        cloned.elseBody = Block()
        for op in self.ifBody.ops:
            op.clone(cloned.ifBody, mapping)
        for op in self.elseBody.ops:
            op.clone(cloned.elseBody, mapping)
        return cloned
