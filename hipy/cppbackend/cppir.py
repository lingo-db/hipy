import textwrap
from abc import abstractmethod
from typing import Dict

import hipy.ir as ir
from hipy.ir import SSAValue
from hipy.cppbackend import CPPBackend, get_column_builder, get_column_accessor


class CppOp(ir.Operation):
    def __init__(self, block, name, args, result):
        block.ops.append(self)
        self.name = name
        self.args = args
        self.result = result

    @abstractmethod
    def produce(self, backend: CPPBackend):
        pass

    def __str__(self):
        return f"{self.result} = {self.name}({', '.join([str(a) for a in self.args])})"

    def get_used_values(self):
        return self.args

    def get_nested_blocks(self):
        return []

    def has_side_effects(self):
        return True

    def serialize(self):
        raise NotImplementedError()

    def replace_uses(self, old: SSAValue, new: SSAValue):
        for i, arg in enumerate(self.args):
            if arg == old:
                self.args[i] = new
        if self.result == old:
            self.result = new
        for b in self.get_nested_blocks():
            for op in b.ops:
                op.replace_uses(old, new)
        return self

    def get_produced_values(self):
        return [self.result]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        return CppOp(self.fn, [mapping[arg] for arg in self.args], mapping[self.result])


class TableBuilderType:
    def __init__(self, members):
        self.members = members

    def __str__(self):
        return f'table_builder[{",".join(map(lambda m: f'{m[0]}: {m[1]}', self.members))}]'



class JoinHtType:
    def __init__(self, key_types, value_types):
        self.key_types = key_types
        self.value_types = value_types

    def __str__(self):
        return f'join_ht[({",".join(map(str, self.key_types))}),({",".join(map(str, self.value_types))})]'

class CreateJoinHt(CppOp):
    def __init__(self, block, join_ht_type):
        self.join_ht_type = join_ht_type

        super().__init__(block, "create_join_ht", [], SSAValue(join_ht_type, self))

    def produce(self, backend: CPPBackend):
        return f"""
          ds::JoinHashTable<std::tuple<{", ".join(map(backend.generate_type, self.join_ht_type.key_types))}>, std::tuple<{", ".join(map(backend.generate_type, self.join_ht_type.value_types))}>> {backend.generate_value(self.result)};
        """


class JoinHtInsert(CppOp):
    def __init__(self, block, join_ht, keys, values):
        self.join_ht = join_ht
        self.keys = keys
        self.values = values

        super().__init__(block, "join_ht_insert", [join_ht, *keys, *values], SSAValue(ir.void,self))

    def produce(self, backend: CPPBackend):
        return f"""
          {backend.generate_value(self.join_ht)}.insert({{ {", ".join([backend.generate_value(key) for key in self.keys])} }}, {{ {", ".join([backend.generate_value(value) for value in self.values])} }});
        """
    def get_used_values(self):
        return self.args

    def replace_uses(self, old: SSAValue, new: SSAValue):
        super().replace_uses(old, new)
        self.keys = [new if v == old else v for v in self.keys]
        self.values = [new if v == old else v for v in self.values]

class JoinHtBuild(CppOp):
    def __init__(self, block, join_ht):
        self.join_ht = join_ht

        super().__init__(block, "join_ht_build", [join_ht], SSAValue(ir.void,self))

    def produce(self, backend: CPPBackend):
        return f"""
          {backend.generate_value(self.join_ht)}.build();
        """


class JoinHtLookup(CppOp):
    def __init__(self, block, join_ht, keys):
        self.join_ht = join_ht
        self.keys = keys

        super().__init__(block, "join_ht_lookup", [join_ht, *keys], SSAValue(ir.void,self))
        self.iter_block = ir.Block()
        self.iter_vars = [ir.SSAValue(t, self) for t in join_ht.type.value_types]

    def produce(self, backend: CPPBackend):
        it_name=f"it{backend.generate_unique_id()}"
        return f"""
          auto {it_name} = {backend.generate_value(self.join_ht)}.find({{ {", ".join([backend.generate_value(key) for key in self.keys])} }});
          for (; {it_name}.valid(); {it_name}.next()) {{
            {"\n".join([f"{backend.generate_result(self.iter_vars[i])} = std::get<{i}>({it_name}.current());" for i in range(len(self.iter_vars))])}
            {"\n".join(map(backend.generate_op, self.iter_block.ops))}
          }}
        """

    def __str__(self):
        return f"join_ht_lookup({self.join_ht},{self.keys}){{\n{textwrap.indent(str(self.iter_block), '  ')}\n}}"

    def get_used_values(self):
        return self.args + ir.flatten([op.get_used_values() for op in self.iter_block.ops])

    def replace_uses(self, old: SSAValue, new: SSAValue):
        super().replace_uses(old,new)
        self.join_ht=self.args[0]
        self.keys=self.args[1:]


    def get_nested_blocks(self):
        return [self.iter_block]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = IterateTable(block, mapping[self.join_ht],[mapping[key] for key in self.keys])
        for op in self.iter_block.ops:
            new_op.iter_block.ops.append(op.clone(new_op.iter_block, mapping))
        return new_op

class AggregationHtType:
    def __init__(self, key_types, value_type):
        self.key_types = key_types
        self.value_type = value_type

    def __str__(self):
        return f'aggregation_ht[({",".join(map(str, self.key_types))}),({self.value_type})]'

class CreateAggregationHt(CppOp):
    def __init__(self, block, aggregation_ht_type):
        self.aggregation_ht_type = aggregation_ht_type

        super().__init__(block, "create_aggregation_ht", [], SSAValue(aggregation_ht_type, self))

    def produce(self, backend: CPPBackend):
        return f"""
          std::unordered_map<std::tuple<{", ".join(map(backend.generate_type, self.aggregation_ht_type.key_types))}>, {backend.generate_type(self.aggregation_ht_type.value_type)}> {backend.generate_value(self.result)};
        """

class Aggregate(CppOp):
    def __init__(self, block, aggregation_ht, keys):
        self.aggregation_ht = aggregation_ht
        self.keys = keys

        super().__init__(block, "aggregate", [aggregation_ht, *keys], SSAValue(ir.void,self))

        self.init_block = ir.Block()
        self.agg_block=ir.Block()
        self.agg_val=ir.SSAValue(aggregation_ht.type.value_type,self)

    def produce(self, backend: CPPBackend):
        it_name = f"it{backend.generate_unique_id()}"
        return f"""
          auto {it_name} = {backend.generate_value(self.aggregation_ht)}.find({{ {", ".join([backend.generate_value(key) for key in self.keys])} }});
          {backend.generate_result(self.agg_val)};
          if ({it_name} == {backend.generate_value(self.aggregation_ht)}.end()) {{
            {"\n".join(map(backend.generate_op, self.init_block.ops[:-1]))}
            {backend.generate_value(self.agg_val)} = {backend.generate_value(self.init_block.ops[-1].values[0])};
            {backend.generate_value(self.aggregation_ht)}.insert({{ {{ {", ".join([backend.generate_value(key) for key in self.keys])} }}, {backend.generate_value(self.agg_val)} }});
            {it_name} = {backend.generate_value(self.aggregation_ht)}.find({{ {", ".join([backend.generate_value(key) for key in self.keys])} }});
          }} else {{
            {backend.generate_value(self.agg_val)} = {it_name}->second;
          }}
          {"\n".join(map(backend.generate_op, self.agg_block.ops[:-1]))}
          {it_name}->second={backend.generate_value(self.agg_block.ops[-1].values[0])};
        """
    def __str__(self):
        return f"aggregate({self.aggregation_ht},{self.keys}){{\n{textwrap.indent(str(self.agg_block), '  ')}\n}}"

    def get_used_values(self):
        return self.args + ir.flatten([op.get_used_values() for op in self.agg_block.ops])

    def replace_uses(self, old: SSAValue, new: SSAValue):
        for i, arg in enumerate(self.args):
            if arg == old:
                self.args[i] = new
        for op in self.agg_block.ops:
            op.replace_uses(old, new)
        return self

    def get_nested_blocks(self):
        return [self.agg_block,self.init_block]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = Aggregate(block, mapping[self.aggregation_ht],[mapping[key] for key in self.keys])
        for op in self.agg_block.ops:
            new_op.agg_block.ops.append(op.clone(new_op.agg_block, mapping))
        for op in self.init_block.ops:
            new_op.init_block.ops.append(op.clone(new_op.init_block, mapping))
        return new_op


class IterateAggregationHt(CppOp):
    def __init__(self, block, aggregation_ht):
        self.aggregation_ht = aggregation_ht

        super().__init__(block, "iterate_aggregation_ht", [aggregation_ht], SSAValue(ir.void,self))
        self.iter_block = ir.Block()
        self.iter_key_vars = [ir.SSAValue(t, self) for t in aggregation_ht.type.key_types]
        self.iter_val = ir.SSAValue(aggregation_ht.type.value_type,self)

    def produce(self, backend: CPPBackend):
        return f"""
          for (const auto& [key, value] : {backend.generate_value(self.aggregation_ht)}){{
            {"\n".join([f"{backend.generate_result(self.iter_key_vars[i])} = std::get<{i}>(key);" for i in range(len(self.iter_key_vars))])}
            {backend.generate_result(self.iter_val)} = value;
            {"\n".join(map(backend.generate_op, self.iter_block.ops))}
          }}
        """

    def __str__(self):
        return f"iterate_aggregation_ht({self.aggregation_ht}){{\n{textwrap.indent(str(self.iter_block), '  ')}\n}}"
    def get_used_values(self):
        return self.args + ir.flatten([op.get_used_values() for op in self.iter_block.ops])
    def replace_uses(self, old: SSAValue, new: SSAValue):
        for i, arg in enumerate(self.args):
            if arg == old:
                self.args[i] = new
        for op in self.iter_block.ops:
            op.replace_uses(old, new)
        return self
    def get_nested_blocks(self):
        return [self.iter_block]
    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = IterateAggregationHt(block, mapping[self.aggregation_ht])
        for op in self.iter_block.ops:
            new_op.iter_block.ops.append(op.clone(new_op.iter_block, mapping))
        return new_op

class FlagType:
    def __init__(self):
        pass

    def __str__(self):
        return 'flag'


class CounterType:
    def __init__(self):
        pass

    def __str__(self):
        return 'counter'


class CreateFlag(CppOp):
    def __init__(self, block, flag_type):
        self.flag_type = flag_type

        super().__init__(block, "create_flag", [], SSAValue(flag_type, self))

    def produce(self, backend: CPPBackend):
        return f"""
          bool {backend.generate_value(self.result)} = false;
        """

class CreateCounter(CppOp):
    def __init__(self, block, counter_type):
        self.counter_type = counter_type

        super().__init__(block, "create_counter", [], SSAValue(counter_type, self))

    def produce(self, backend: CPPBackend):
        return f"""
          int64_t {backend.generate_value(self.result)} = 0;
        """


class SetFlag(CppOp):
    def __init__(self, block, flag):
        self.flag = flag

        super().__init__(block, "set_flag", [flag], SSAValue(ir.void,self))

    def produce(self, backend: CPPBackend):
        return f"""
          {backend.generate_value(self.flag)} = true;
        """

class IncrementCounter(CppOp):
    def __init__(self, block, counter):
        self.counter = counter

        super().__init__(block, "increment_counter", [counter], SSAValue(ir.i64,self))

    def produce(self, backend: CPPBackend):
        return f"""
          {backend.generate_result(self.result)} = {backend.generate_value(self.counter)} ++;
        """

class CheckFlag(CppOp):
    def __init__(self, block, flag):
        self.flag = flag

        super().__init__(block, "check_flag", [flag], SSAValue(ir.bool,self))

    def produce(self, backend: CPPBackend):
        return f"""
          {backend.generate_result(self.result)} = {backend.generate_value(self.flag)};
        """

class CreateTableBuilder(CppOp):
    def __init__(self, block, table_builder_type):
        self.builder_vars = []
        self.build_cols = []
        self.table_builder_type = table_builder_type

        super().__init__(block, "create_table_builder", [], SSAValue(table_builder_type, self))

    def produce(self, backend: CPPBackend):
        res = ""
        for col, t in self.table_builder_type.members:
            builder_var = f"col_builder{backend.generate_unique_id()}"
            self.builder_vars.append(builder_var)
            res += (f"{get_column_builder(t)} {builder_var};\n")
        return res




class IterateTable(CppOp):
    def __init__(self, block, table, required_cols):
        self.table = table
        required_cols=[col for col,_ in table.type.members if col in required_cols]
        self.required_cols = required_cols

        super().__init__(block, "iterate_table", [table], SSAValue(ir.void,self))
        self.iter_block = ir.Block()
        self.iter_vars = [ir.SSAValue(t, self) for n, t in table.type.members if n in required_cols]
        #print(self.iter_vars)

    def produce(self, backend: CPPBackend):
        accessor_vars = {col: f"col_accessor{backend.generate_unique_id()}" for col in self.required_cols}
        arg_col_types = {col: type for col, type in self.table.type.members}
        arg_col_offsets = {col: offset for offset, (col, _) in enumerate(self.table.type.members)}

        return f"""
        {backend.generate_value(self.table)}->iterateBatches([&](auto batch){{
        {"\n".join([f"auto {accessor_vars[col]} = {get_column_accessor(arg_col_types[col])}(batch->column({arg_col_offsets[col]}));" for col in self.required_cols])}
        for (int64_t i = 0; i < batch->num_rows(); i++){{
            {"\n".join([f"{backend.generate_result(self.iter_vars[i])} ={accessor_vars[col]}.access(i);" for i,col in enumerate(self.required_cols)])}
            {"\n".join(map(backend.generate_op, self.iter_block.ops))}
            }}
        }});
        """

    def get_used_values(self):
        return self.args + ir.flatten([op.get_used_values() for op in self.iter_block.ops])

    def replace_uses(self, old: SSAValue, new: SSAValue):
        for i, arg in enumerate(self.args):
            if arg == old:
                self.args[i] = new
        for op in self.iter_block.ops:
            op.replace_uses(old, new)
        return self

    def get_nested_blocks(self):
        return [self.iter_block]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = IterateTable(block, mapping[self.table])
        for op in self.iter_block.ops:
            new_op.iter_block.ops.append(op.clone(new_op.iter_block, mapping))
        return new_op

    def __str__(self):
        return f"iterate_table({self.table}){{\n{textwrap.indent(str(self.iter_block), '  ')}\n}}"


class TableBuilderAppend(CppOp):
    def __init__(self, block, table_builder, values):
        self.table_builder = table_builder
        self.values = values

        super().__init__(block, "table_builder_append", [table_builder],
                         SSAValue(ir.void,self))

    def produce(self, backend: CPPBackend):
        match self.table_builder.producer:
            case CreateTableBuilder(builder_vars=builder_vars):
                return "\n".join([f"{builder_var}.append({backend.generate_value(self.values[value])});" for builder_var, value in zip(builder_vars, self.values)])
            case _:
                raise ValueError("TableBuilderAppend can only be used with CreateTableBuilder")

    def get_used_values(self):
        return self.args + list(self.values.values())
    def replace_uses(self, old: SSAValue, new: SSAValue):
        super().replace_uses(old, new)
        self.values = {k: new if v == old else v for k, v in self.values.items()}
    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = TableBuilderAppend(block, mapping[self.table_builder], {k: mapping[v] for k, v in self.values.items()})
        mapping[self.result] = new_op.result
        return new_op



class TableBuilderFinish(CppOp):
    def __init__(self, block, table_builder, res_table_type):
        self.table_builder = table_builder
        self.table_type = res_table_type
        super().__init__(block, "table_builder_finish", [table_builder], SSAValue(res_table_type,self))

    def produce(self, backend: CPPBackend):
        match self.table_builder.producer:
            case CreateTableBuilder(builder_vars=builder_vars):
                builder_vars={col:var for col,var in zip([col for col,t in self.table_builder.type.members],builder_vars)}
                computed_cols=[col for col,t in self.table_builder.type.members]
                column_vars={col: f"built_column_{backend.generate_unique_id()}" for col in computed_cols}
                return f"""{"\n".join([f"auto {column_vars[col]} = {builder_vars[col]}.build();" for col in computed_cols])}
                {backend.generate_result(self.result)} =builtin::tabular::Table::from_columns({{ {",".join([f"{{\"{col}\",{column_vars[col]} }}" for col in computed_cols])} }});
                """

            case _:
                raise ValueError("TableBuilderFinish can only be used with CreateTableBuilder")


class IterRange(CppOp):
    def __init__(self, block,initial_values,res_type,start, end, step):
        self.initial_values = initial_values
        self.start = start
        self.end = end
        self.step = step
        self.res_type=res_type
        super().__init__(block, "iter_range", [start, end, step], SSAValue(res_type,self))
        self.iter_block = ir.Block()
        self.finish_block = ir.Block()
        self.iter_var = ir.SSAValue(ir.i64,self)
        self.iter_val_vars= {k: ir.SSAValue(v.type,self) for k,v in initial_values.items()}

    def produce(self, backend: CPPBackend):
        return f"""
        {"\n".join([f"{backend.generate_result(self.iter_val_vars[k])} = {backend.generate_value(v)};" for k,v in self.initial_values.items()])}
        for (int64_t {backend.generate_value(self.iter_var)} = {backend.generate_value(self.start)};  {backend.generate_value(self.step)}<0 ? ({backend.generate_value(self.iter_var)}>{backend.generate_value(self.end)}):({backend.generate_value(self.iter_var)}<{backend.generate_value(self.end)}); {backend.generate_value(self.iter_var)} += {backend.generate_value(self.step)}){{
            {"\n".join(map(backend.generate_op, self.iter_block.ops[:-1]))}
            {"\n".join([f"{backend.generate_value(self.iter_val_vars[k])} = {backend.generate_value(v)};" for k,v in zip(self.iter_val_vars,self.iter_block.ops[-1].values)])}
        }}
        
            {backend.generate_result(self.result)} = std::make_tuple({", ".join([backend.generate_value(v) for v in self.iter_val_vars.values()])});
        """

    def get_used_values(self):
        r= list(self.initial_values.values())+ self.args + ir.flatten([op.get_used_values() for op in self.iter_block.ops])
        for x in r:
            assert isinstance(x,SSAValue)
        return r

    def replace_uses(self, old: SSAValue, new: SSAValue):
        super().replace_uses(old, new)
        self.initial_values = {k: new if v == old else v for k, v in self.initial_values.items()}
        self.start=self.args[0]
        self.end=self.args[1]
        self.step=self.args[2]

    def __str__(self):
        return f"iter_range({self.initial_values},{self.start},{self.end},{self.step}){{\n{textwrap.indent(str(self.iter_block), '  ')}\n}}"

    def get_nested_blocks(self):
        return [self.iter_block]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = IterRange(block,{k:mapping[v] for k,v in self.initial_values.items()},self.res_type,mapping[self.start],mapping[self.end],mapping[self.step])
        mapping[self.iter_var]=new_op.iter_var
        for k,v in self.iter_val_vars.items():
            mapping[v]=new_op.iter_val_vars[k]
        for op in self.iter_block.ops:
           op.clone(new_op.iter_block, mapping)
        mapping[self.result]=new_op.result
        return new_op


class IterRange_(CppOp):
    def __init__(self, block, initial_value, start, end, step):
        self.initial_value = initial_value
        self.start = start
        self.end = end
        self.step = step

        super().__init__(block, "iter_range", [initial_value, start, end, step], SSAValue(initial_value.type, self))
        self.iter_block = ir.Block()
        self.finish_block = ir.Block()
        self.iter_var = ir.SSAValue(ir.i64, self)
        self.iter_val_var = ir.SSAValue(initial_value.type, self)

    def produce(self, backend: CPPBackend):
        return f"""
        {backend.generate_result(self.result)} = {backend.generate_value(self.initial_value)};

        for (int64_t {backend.generate_value(self.iter_var)} = {backend.generate_value(self.start)}; {backend.generate_value(self.iter_var)} < {backend.generate_value(self.end)}; {backend.generate_value(self.iter_var)} += {backend.generate_value(self.step)}){{
            {backend.generate_result(self.iter_val_var)} = {backend.generate_value(self.result)};
            {"\n".join(map(backend.generate_op, self.iter_block.ops[:-1]))}
            {backend.generate_result(self.result)} = {backend.generate_value(self.iter_block.ops[-1].values[0])};
        }}
        """

    def get_used_values(self):
        return self.args + ir.flatten([op.get_used_values() for op in self.iter_block.ops])

    def replace_uses(self, old: SSAValue, new: SSAValue):
        super().replace_uses(old, new)
        self.initial_value = self.args[0]
        self.start = self.args[1]
        self.end = self.args[2]
        self.step = self.args[3]

    def __str__(self):
        return f"iter_range({self.initial_value},{self.start},{self.end},{self.step}){{\n{textwrap.indent(str(self.iter_block), '  ')}\n}}"

    def get_nested_blocks(self):
        return [self.iter_block]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = IterRange(block, mapping[self.initial_value], mapping[self.start], mapping[self.end],
                           mapping[self.step])
        mapping[self.iter_var] = new_op.iter_var
        mapping[self.iter_val_var] = new_op.iter_val_var
        for op in self.iter_block.ops:
            op.clone(new_op.iter_block, mapping)
        mapping[self.result] = new_op.result
        return new_op

class WhileIter(CppOp):
    def __init__(self, block, read_only,initial_value):
        self.initial_value = initial_value
        self.read_only = read_only

        super().__init__(block, "while_iter", [initial_value, read_only], SSAValue(initial_value.type,self))
        self.cond_block = ir.Block()
        self.iter_block = ir.Block()
        self.iter_val_var=ir.SSAValue(initial_value.type,self)



    def produce(self, backend: CPPBackend):
        return f"""
        {backend.generate_result(self.result)} = {backend.generate_value(self.initial_value)};
        
        while (true){{
            {backend.generate_result(self.iter_val_var)} = {backend.generate_value(self.result)};
            {"\n".join(map(backend.generate_op, self.cond_block.ops[:-1]))}
            if (!{backend.generate_value(self.cond_block.ops[-1].values[0])}){{
                break;
            }}
            {"\n".join(map(backend.generate_op, self.iter_block.ops[:-1]))}
            {backend.generate_value(self.result)} = {backend.generate_value(self.iter_block.ops[-1].values[0])};
        }}
        """

    def get_nested_blocks(self):
        return [self.cond_block,self.iter_block]
    def __str__(self):
        return f"while_iter({self.read_only},{self.initial_value}) {{\n{textwrap.indent(str(self.cond_block), '  ')}\n}} {{\n{textwrap.indent(str(self.iter_block), '  ')}\n}}"
    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op = WhileIter(block, mapping[self.read_only],mapping[self.initial_value])
        mapping[self.iter_val_var]=new_op.iter_val_var
        mapping[self.result]=new_op.result
        for op in self.cond_block.ops:
            op.clone(new_op.cond_block, mapping)
        for op in self.iter_block.ops:
            op.clone(new_op.iter_block, mapping)
        return new_op

    def replace_uses(self, old: SSAValue, new: SSAValue):
        super().replace_uses(old, new)
        self.initial_value=self.args[0]
        self.read_only=self.args[1]


class PyMethodCall(CppOp):
    def __init__(self, block, obj, method, args):
        self.obj = obj
        self.method = method
        self.args = args

        super().__init__(block, "py_method_call", [obj, method, *args], SSAValue(ir.pyobj,self))

    def produce(self, backend: CPPBackend):
        return f"""
        {backend.generate_result(self.result)} =py::reinterpret_steal<py::object>(
                    PyObject_CallMethodObjArgs({backend.generate_value(self.obj)}.ptr(),{backend.generate_value(self.method)}.ptr(), {",".join([f"{backend.generate_value(v)}.ptr()"  for v in self.args])},NULL));
        """

    def replace_uses(self, old: SSAValue, new: SSAValue):
        super().replace_uses(old, new)
        self.obj=self.args[0]
        self.method=self.args[1]
        self.args=self.args[2:]

    def clone(self, block, mapping: Dict[SSAValue, SSAValue]):
        new_op=PyMethodCall(block, mapping[self.obj],mapping[self.method],[mapping[arg] for arg in self.args])
        mapping[self.result]=new_op.result
        return new_op