import os
import pathlib
import sys
import textwrap
from typing import Any, Dict

from jinja2 import Template, FileSystemLoader, Environment
from hipy import ir

current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the template folder relative to the current module's directory
template_dir = os.path.join(current_dir, 'templates')
template_loader = FileSystemLoader(searchpath=template_dir)
env = Environment(loader=template_loader)

import subprocess


def write_compile_run_cpp(code, release=False):
    standalone_project_path = f"{pathlib.Path(__file__).parent.resolve()}/../../cppbackend"
    standalone_build_path = f"{standalone_project_path}/build-debug" if not release else f"{standalone_project_path}/build-release"
    if "HIPY_STANDALONE_SOURCE" in os.environ:
        standalone_project_path = os.environ["HIPY_STANDALONE_SOURCE"]
    if "HIPY_STANDALONE_BUILD" in os.environ:
        standalone_build_path = os.environ["HIPY_STANDALONE_BUILD"]

    # Write the C++ code to a file
    with open(f'{standalone_project_path}/standalone.cpp', 'w') as file:
        file.write(code)

    cmake_command = ['cmake', '--build', standalone_build_path, '--target', 'standalone']
    cmake_process = subprocess.Popen(cmake_command, stderr=subprocess.PIPE)
    cmake_output, cmake_error = cmake_process.communicate()

    if cmake_process.returncode != 0:
        return (None, cmake_error.decode('utf-8'), cmake_process.returncode)

    # Run the compiled binary
    print(f"env PYTHONPATH={':'.join(sys.path)} {standalone_build_path}/standalone")
    run_command = [f'{standalone_build_path}/standalone']
    run_process = subprocess.Popen(run_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   env={"PYTHONPATH": ':'.join(sys.path)})
    run_output, run_error = run_process.communicate()

    return (run_output.decode('utf-8'), run_error.decode('utf-8'), run_process.returncode)


def get_column_builder(t):
    match t:
        case ir.BoolType():
            return "builtin::tabular::BoolColumnBuilder"
        case ir.IntType():
            return "builtin::tabular::Int64ColumnBuilder"
        case ir.IntegerType(width=w):
            match w:
                case 8:
                    return "builtin::tabular::Int8ColumnBuilder"
                case 16:
                    return "builtin::tabular::Int16ColumnBuilder"
                case 32:
                    return "builtin::tabular::Int32ColumnBuilder"
                case 64:
                    return "builtin::tabular::Int64ColumnBuilder"
        case ir.FloatType(width=w):
            match w:
                case 32:
                    return "builtin::tabular::Float32ColumnBuilder"
                case 64:
                    return "builtin::tabular::Float64ColumnBuilder"
                case _:
                    raise NotImplementedError(f"Float width {w} not implemented")
        case ir.StringType():
            return "builtin::tabular::StrColumnBuilder"
        case ir.ListType(element_type=element_type):
            return f"builtin::tabular::ListColumnBuilder<{get_column_builder(element_type)}>"
        case _:
            raise NotImplementedError(f"Type {t} not implemented")


def get_column_accessor(t):
    match t:
        case ir.BoolType():
            return "builtin::tabular::BoolColumnAccessor"
        case ir.IntType():
            return "builtin::tabular::Int64ColumnAccessor"
        case ir.IntegerType(width=w):
            match w:
                case 8:
                    return "builtin::tabular::Int8ColumnAccessor"
                case 16:
                    return "builtin::tabular::Int16ColumnAccessor"
                case 32:
                    return "builtin::tabular::Int32ColumnAccessor"
                case 64:
                    return "builtin::tabular::Int64ColumnAccessor"
        case ir.FloatType(width=w):
            match w:
                case 32:
                    return "builtin::tabular::Float32ColumnAccessor"
                case 64:
                    return "builtin::tabular::Float64ColumnAccessor"
                case _:
                    raise NotImplementedError(f"Float width {w} not implemented")
        case ir.StringType():
            return "builtin::tabular::StrColumnAccessor"
        case ir.ListType(element_type=element_type):
            return f"builtin::tabular::ListColumnAccessor<{get_column_accessor(element_type)}>"
        case _:
            raise NotImplementedError(f"Type {t} not implemented")


class CPPBackend:
    def __init__(self, fn_name, module: ir.Module, release=False):
        self.unique_id = 0
        self.release = release
        self.fn_name = fn_name
        self.module = module
        self.enable_python = False
        self.enable_arrow = False
        self.enable_numpy = False
        self.global_constants = ""
        self.global_py_constants = {}

    def generate_type(self, t):
        match t:
            case ir.VoidType():
                return "uint8_t"
            case ir.StringType():
                return "std::string"
            case ir.IntType():
                return "int64_t"  # todo
            case ir.IntegerType(width=w):
                match w:
                    case 8:
                        return "int8_t"
                    case 16:
                        return "int16_t"
                    case 32:
                        return "int32_t"
                    case 64:
                        return "int64_t"
                    case _:
                        raise NotImplementedError(f"Int width {w} not implemented")
            case ir.BoolType():
                return "bool"
            case ir.PyObjType():
                self.enable_python |= True
                return "py::object"
            case ir.FloatType(width=w):
                match w:
                    case 32:
                        return "float"
                    case 64:
                        return "double"
                    case _:
                        raise NotImplementedError(f"Float width {w} not implemented")
            case ir.ListType(element_type=element_type):
                return f"std::shared_ptr<std::vector<{self.generate_type(element_type)}>>"
            case ir.DictType(key_type=key_type, val_type=value_type):
                return f"std::shared_ptr<std::unordered_map<{self.generate_type(key_type)},{self.generate_type(value_type)}>>"
            case ir.RecordType(members=members):
                return "std::tuple<" + ", ".join(map(lambda m: self.generate_type(m[1]), members)) + ">"
            case ir.ArrayType(element_type=element_type, shape=shape):
                self.enable_numpy |= True
                num_dimensions = len(shape)
                return f"std::shared_ptr<builtin::ndarray<{self.generate_type(element_type)},{num_dimensions}>>"
            case ir.ColumnType():
                self.enable_arrow |= True
                return f"std::shared_ptr<builtin::tabular::Column>"
            case ir.TableType():
                self.enable_arrow |= True
                return f"std::shared_ptr<builtin::tabular::Table>"
            case ir.FunctionRefType(arg_types=arg_types, res_type=res_type, closure_type=closure_type):
                if closure_type is None:
                    return f"std::add_pointer<{self.generate_type(res_type)}({', '.join(map(self.generate_type, arg_types))})>::type"
                else:
                    return f"builtin::bound_fn<std::add_pointer<{self.generate_type(res_type)}({', '.join(map(self.generate_type, arg_types))})>::type, {self.generate_type(closure_type)}>"
            case _:
                raise NotImplementedError(f"Type {t} not implemented")

    def generate_value(self, val: ir.SSAValue):
        return f"val_{val.id}"

    def escape_for_cpp(self, string):
        escaped_string = string.replace('\\', '\\\\')
        escaped_string = escaped_string.replace('\n', '\\n')
        escaped_string = escaped_string.replace('\t', '\\t')
        escaped_string = escaped_string.replace('\r', '\\r')
        escaped_string = escaped_string.replace('"', '\\"')
        return escaped_string

    def generate_constant(self, v):
        match v:
            case None:
                return 0

            case bool():
                return "true" if v else "false"
            case int() | float():
                if f"{v}"=="nan":
                    return "NAN"
                return f"{v}"
            case str():
                escaped = self.escape_for_cpp(v)
                return f"std::string(\"{escaped}\",{len(v)})"
            case _:
                raise NotImplementedError()

    def generate_result(self, val: ir.SSAValue):
        return f"{self.generate_type(val.type)} {self.generate_value(val)}"

    def generate_unique_id(self):
        self.unique_id += 1
        return self.unique_id

    def generate_builtin(self, op: ir.CallBuiltin):
        def binary_op(fn):
            return f"{self.generate_result(op.result)} = {fn(self.generate_value(op.args[0]), self.generate_value(op.args[1]))};"

        def py_cmp_op(cmp_type):
            return f"{self.generate_result(op.result)} = py::reinterpret_steal<py::object>(PyObject_RichCompare({self.generate_value(op.args[0])}.ptr(),{self.generate_value(op.args[1])}.ptr(), {cmp_type}));"

        def iter_rec(array, D, d, indices, fn):
            if d == D:
                return fn(indices)
            else:
                return f"""
    for (int64_t i{d} = 0; i{d} < {self.generate_value(array)}->getDimension({d}); i{d}++){{
        {iter_rec(array, D, d + 1, indices + [f"i{d}"], fn)}
    }}
    """

        def call(func, arg_str):
            match func.producer:
                case ir.FunctionRef(name=fn_name, closure=closure):
                    if closure is None:
                        return f"{fn_name}({arg_str})"
                    else:
                        return f"{fn_name}({arg_str}{"," if arg_str else ""}{self.generate_value(closure)})"
                case _:
                    return self.generate_value(func)

        match op.name:
            case "dbg.print":
                return f"builtin::print({self.generate_value(op.args[0])});"
            case "scalar.int.add":
                return binary_op(lambda x, y: f"{x}+{y}")
            case "scalar.int.sub":
                return binary_op(lambda x, y: f"{x}-{y}")
            case "scalar.int.mul":
                return binary_op(lambda x, y: f"{x}*{y}")
            case "scalar.int.div":
                return binary_op(lambda x, y: f"{x}/{y}")
            case "scalar.int.mod":
                return binary_op(lambda x, y: f"{x}%{y}")
            case "scalar.int.compare.lt":
                return binary_op(lambda x, y: f"{x}<{y}")
            case "scalar.int.compare.gt":
                return binary_op(lambda x, y: f"{x}>{y}")
            case "scalar.int.compare.eq":
                return binary_op(lambda x, y: f"{x}=={y}")
            case "scalar.int.compare.neq":
                return binary_op(lambda x, y: f"{x}!={y}")
            case "scalar.int.compare.lte":
                return binary_op(lambda x, y: f"{x}<={y}")
            case "scalar.int.compare.gte":
                return binary_op(lambda x, y: f"{x}>={y}")
            case "scalar.float.add":
                return binary_op(lambda x, y: f"{x}+{y}")
            case "scalar.float.sub":
                return binary_op(lambda x, y: f"{x}-{y}")
            case "scalar.float.mul":
                return binary_op(lambda x, y: f"{x}*{y}")
            case "scalar.float.div":
                return binary_op(lambda x, y: f"{x}/{y}")
            case "scalar.float.compare.lt":
                return binary_op(lambda x, y: f"{x}<{y}")
            case "scalar.float.compare.gt":
                return binary_op(lambda x, y: f"{x}>{y}")
            case "scalar.float.compare.eq":
                return binary_op(lambda x, y: f"{x}=={y}")
            case "scalar.float.compare.neq":
                return binary_op(lambda x, y: f"{x}!={y}")
            case "scalar.float.compare.lte":
                return binary_op(lambda x, y: f"{x}<={y}")
            case "scalar.float.compare.gte":
                return binary_op(lambda x, y: f"{x}>={y}")
            case "scalar.bool.and":
                return binary_op(lambda x, y: f"{x}&&{y}")
            case "scalar.bool.or":
                return binary_op(lambda x, y: f"{x}||{y}")
            case "scalar.bool.not":
                return f"{self.generate_result(op.result)} = !{self.generate_value(op.args[0])};"
            case "scalar.string.ord":
                return f"{self.generate_result(op.result)} = (int64_t)({self.generate_value(op.args[0])}[0]);"
            case "scalar.string.concatenate":
                return binary_op(lambda x, y: f"{x}+{y}")
            case "scalar.int.to_string":
                return f"{self.generate_result(op.result)} =  std::to_string({self.generate_value(op.args[0])});"
            case "scalar.int.from_string":
                return f"{self.generate_result(op.result)} =  std::stoll({self.generate_value(op.args[0])});"
            case "scalar.bool.to_string":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])} ? \"True\" : \"False\";"
            case "scalar.bool.to_python":
                return f"{self.generate_result(op.result)} =  py::bool_({self.generate_value(op.args[0])});"
            case "scalar.bool.from_python":
                return f"{self.generate_result(op.result)} =  (bool)PyObject_IsTrue({self.generate_value(op.args[0])}.ptr());"
            case "scalar.float.to_string":
                return f"{self.generate_result(op.result)} =  builtin::float_to_string({self.generate_value(op.args[0])});"
            case "scalar.float.to_python":
                match op.args[0].producer:
                    case ir.Constant(v=v):
                        self.global_py_constants[self.generate_value(op.result)] = f"py::float_({v});"
                        return ""
                    case _:
                        return f"{self.generate_result(op.result)} =  py::float_({self.generate_value(op.args[0])});"
            case "scalar.float.from_python":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])}.cast<{self.generate_type(op.result.type)}>();"
            case "scalar.float.from_int":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])};"
            case "scalar.float.to_int":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])};"
            case "scalar.int.pyint_to_int64":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])};"
            case "scalar.int.int64_to_pyint":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])};"
            case "scalar.int.to_python":
                match op.args[0].producer:
                    case ir.Constant(v=v):
                        self.global_py_constants[self.generate_value(op.result)] = f"py::int_({v});"
                        return ""
                    case _:
                        return f"{self.generate_result(op.result)} =  py::int_({self.generate_value(op.args[0])});"
            case "scalar.int.from_python":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])}.cast<{self.generate_type(op.result.type)}>();"
            case "scalar.float.sqrt":
                return f"{self.generate_result(op.result)} =  std::sqrt({self.generate_value(op.args[0])});"
            case "scalar.float.log":
                return f"{self.generate_result(op.result)} =  std::log({self.generate_value(op.args[0])});"
            case "scalar.float.exp":
                return f"{self.generate_result(op.result)} =  std::exp({self.generate_value(op.args[0])});"
            case "scalar.float.erf":
                return f"{self.generate_result(op.result)} =  std::erf({self.generate_value(op.args[0])});"
            case "scalar.float.sin":
                return f"{self.generate_result(op.result)} =  std::sin({self.generate_value(op.args[0])});"
            case "scalar.float.cos":
                return f"{self.generate_result(op.result)} =  std::cos({self.generate_value(op.args[0])});"
            case "scalar.float.arcsin":
                return f"{self.generate_result(op.result)} =  std::asin({self.generate_value(op.args[0])});"
            case "scalar.float.pow":
                return f"{self.generate_result(op.result)} =  std::pow({self.generate_value(op.args[0])},{self.generate_value(op.args[1])});"
            case "random.rand":
                return f"{self.generate_result(op.result)} =  drand48();"

            case "scalar.string.to_python":
                match op.args[0].producer:
                    case ir.Constant(v=v):
                        self.global_py_constants[self.generate_value(op.result)] = f"py::str(\"{self.escape_for_cpp(v)}\");"
                        return ""
                    case _:
                        return f"{self.generate_result(op.result)} =  py::str({self.generate_value(op.args[0])});"
            case "scalar.bytes.to_python":
                return f"{self.generate_result(op.result)} =  py::bytes({self.generate_value(op.args[0])});"
            case "scalar.string.from_python":
                return f"{self.generate_result(op.result)} =  {self.generate_value(op.args[0])}.cast<std::string>();"
            case "scalar.string.strip":
                return f"{self.generate_result(op.result)} =  builtin::string::strip({self.generate_value(op.args[0])});"
            case "scalar.string.rstrip":
                return f"{self.generate_result(op.result)} =  builtin::string::rstrip({self.generate_value(op.args[0])});"
            case "scalar.string.split":
                return f"{self.generate_result(op.result)} =  builtin::string::split({self.generate_value(op.args[0])},{self.generate_value(op.args[1])},{self.generate_value(op.args[2])});"
            case "scalar.string.lower":
                return f"{self.generate_result(op.result)} =  builtin::string::lower({self.generate_value(op.args[0])});"
            case "scalar.string.upper":
                return f"{self.generate_result(op.result)} =  builtin::string::upper({self.generate_value(op.args[0])});"
            case "scalar.string.substr":
                return f"{self.generate_result(op.result)} =  builtin::string::substr({self.generate_value(op.args[0])},{self.generate_value(op.args[1])},{self.generate_value(op.args[2])});"
            case "scalar.string.contains":
                return f"{self.generate_result(op.result)} =  builtin::string::contains({self.generate_value(op.args[0])},{self.generate_value(op.args[1])});"
            case "scalar.string.compare.eq":
                return binary_op(lambda x, y: f"{x}=={y}")
            case "scalar.string.compare.neq":
                return binary_op(lambda x, y: f"{x}!={y}")
            case "scalar.string.compare.lt":
                return binary_op(lambda x, y: f"{x}<{y}")
            case "scalar.string.compare.gt":
                return binary_op(lambda x, y: f"{x}>{y}")
            case "scalar.string.compare.lte":
                return binary_op(lambda x, y: f"{x}<={y}")
            case "scalar.string.compare.gte":
                return binary_op(lambda x, y: f"{x}>={y}")
            case "scalar.string.at":
                return f"{self.generate_result(op.result)} =  builtin::string::at({self.generate_value(op.args[0])},{self.generate_value(op.args[1])});"
            case "scalar.string.length":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}.size();"
            case "scalar.string.find":
                return f"{self.generate_result(op.result)} =  builtin::string::find({self.generate_value(op.args[0])},{self.generate_value(op.args[1])},{self.generate_value(op.args[2])},{self.generate_value(op.args[3])});"
            case "scalar.string.rfind":
                return f"{self.generate_result(op.result)} =  builtin::string::rfind({self.generate_value(op.args[0])},{self.generate_value(op.args[1])},{self.generate_value(op.args[2])},{self.generate_value(op.args[3])});"
            case "scalar.string.replace":
                return f"{self.generate_result(op.result)} =  builtin::string::replace({self.generate_value(op.args[0])},{self.generate_value(op.args[1])},{self.generate_value(op.args[2])});"
            case "list.create":
                return f"{self.generate_result(op.result)} = std::make_shared<std::vector<{self.generate_type(op.result.type.element_type)}>>();"
            case "list.append":
                return f"{self.generate_value(op.args[0])}->push_back({self.generate_value(op.args[1])});"
            case "list.iter":
                iter_var = f"v{self.generate_unique_id()}"
                return f"""
    {self.generate_result(op.result)} = {self.generate_value(op.args[2])};
    for (auto& {iter_var}: *{self.generate_value(op.args[3])}){{
        {self.generate_value(op.result)} = {self.generate_value(op.args[0])}({self.generate_value(op.args[1])},{self.generate_value(op.result)}, {iter_var});
    }}
                """
            case "scalar.string.iter":
                iter_var = f"v{self.generate_unique_id()}"
                return f"""
            {self.generate_result(op.result)} = {self.generate_value(op.args[2])};
            for (auto {iter_var}=0ull;{iter_var}<{self.generate_value(op.args[3])}.size();{iter_var}++){{
                {self.generate_value(op.result)} = {self.generate_value(op.args[0])}({self.generate_value(op.args[1])},{self.generate_value(op.result)}, std::string(1,{self.generate_value(op.args[3])}[{iter_var}]));
            }}
                        """
            case "list.length":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->size();"
            case "list.at":
                return f"{self.generate_result(op.result)} = (*{self.generate_value(op.args[0])})[{self.generate_value(op.args[1])}];"
            case "list.set":
                return f"(*{self.generate_value(op.args[0])})[{self.generate_value(op.args[1])}] = {self.generate_value(op.args[2])};"
            case "list.sort":
                return f"std::sort({self.generate_value(op.args[0])}->begin(), {self.generate_value(op.args[0])}->end(),[&](auto left, auto right){{return {call(op.args[1],"left,right")};}});"
            case "try_or_default":
                return f"""
    {self.generate_result(op.result)};
    try{{
        {self.generate_value(op.result)} = {self.generate_value(op.args[0])}();
        }} catch (...) {{
        {self.generate_value(op.result)} = {self.generate_value(op.args[1])};
        }}
    """
            case "try_except":
                return f"""
    {self.generate_result(op.result)};
    try{{
        {self.generate_value(op.result)} = {call(op.args[0], "")};
        }} catch (...) {{
        {self.generate_value(op.result)} = {call(op.args[1], "")};
        }}
    """
            case "test.apply":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}({', '.join(map(self.generate_value, op.args[1:]))});"
            case "range.iter":
                start = self.generate_value(op.args[3])
                stop = self.generate_value(op.args[4])
                step = self.generate_value(op.args[5])
                initial_value = self.generate_value(op.args[2])
                read_only = self.generate_value(op.args[1])
                iter_var = f"i{self.generate_unique_id()}"
                return f"""
    {self.generate_result(op.result)} = {initial_value};
    for (int64_t {iter_var} = {start}; {step}<0 ? ({iter_var}>{stop}):({iter_var}<{stop});{iter_var}+={step}){{
        {self.generate_value(op.result)} = {call(op.args[0], f"{read_only},{self.generate_value(op.result)},{iter_var}")};
    }}
    """
            case "while.iter":
                initial_value = self.generate_value(op.args[3])
                read_only = self.generate_value(op.args[2])
                iter_var = f"i{self.generate_unique_id()}"
                return f"""
    {self.generate_result(op.result)} = {initial_value};
    while({call(op.args[0], f"{read_only},{self.generate_value(op.result)}")}){{
        {self.generate_value(op.result)} = {call(op.args[1], f"{read_only},{self.generate_value(op.result)}")};
    }}
    """
            case "python.create_list":
                return f"{self.generate_result(op.result)} = py::list();"
            case "python.get_none":
                return f"{self.generate_result(op.result)} = py::none();"
            case "python.create_slice":
                return f"{self.generate_result(op.result)} = py::slice({self.generate_value(op.args[0])},{self.generate_value(op.args[1])},{self.generate_value(op.args[2])});"
            case "python.create_dict":
                return f"{self.generate_result(op.result)} = py::dict();"
            case "python.tuple_from_list":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}.cast<py::tuple>();"
            case "python.operator.gt":
                return py_cmp_op("Py_GT")
            case "python.operator.lt":
                return py_cmp_op("Py_LT")
            case "python.operator.eq":
                return py_cmp_op("Py_EQ")
            case "python.operator.add":
                return binary_op(lambda x, y: f"{x}+{y}")
            case "python.operator.sub":
                return binary_op(lambda x, y: f"{x}-{y}")
            case "python.operator.mul":
                return binary_op(lambda x, y: f"{x}*{y}")
            case "python.operator.div":
                return binary_op(lambda x, y: f"{x}/{y}")
            case "python.operator.mod":
                return binary_op(lambda x, y: f"{x}%{y}")
            case "python.operator.contains":
                return binary_op(lambda x, y: f"{x}.contains({y})")
            case "dict.create":
                return f"{self.generate_result(op.result)} = std::make_shared<std::unordered_map<{self.generate_type(op.result.type.key_type)},{self.generate_type(op.result.type.val_type)}>>();"
            case "dict.set":
                return f"{self.generate_value(op.args[0])}->operator[]({self.generate_value(op.args[1])}) = {self.generate_value(op.args[2])};"
            case "dict.get":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->operator[]({self.generate_value(op.args[1])});"
            case "dict.contains":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->contains({self.generate_value(op.args[1])});"
            case "dict.iter_keys":
                iter_var = f"v{self.generate_unique_id()}"
                return f"""
            {self.generate_result(op.result)} = {self.generate_value(op.args[2])};
            for (auto& {iter_var}: *{self.generate_value(op.args[3])}){{
                {self.generate_value(op.result)} = {self.generate_value(op.args[0])}({self.generate_value(op.args[1])},{self.generate_value(op.result)}, {iter_var}.first);
            }}
                        """
            case "dict.iter_items":
                iter_var = f"v{self.generate_unique_id()}"
                return f"""
            {self.generate_result(op.result)} = {self.generate_value(op.args[2])};
            for (auto& {iter_var}: *{self.generate_value(op.args[3])}){{
                {self.generate_value(op.result)} = {self.generate_value(op.args[0])}({self.generate_value(op.args[1])},{self.generate_value(op.result)}, std::make_tuple({iter_var}.first, {iter_var}.second));
            }}
                        """
            case "dict.length":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->size();"
            case "array.create_empty":
                num_dimensions = len(op.result.type.shape)
                dims = f"builtin::tupleToArray({self.generate_value(op.args[0])})"
                return f"{self.generate_result(op.result)} = std::make_shared<builtin::ndarray<{self.generate_type(op.result.type.element_type)},{num_dimensions}>>({dims});"
            case "array.reshape":
                num_dimensions = len(op.result.type.shape)
                dims = f"builtin::tupleToArray({self.generate_value(op.args[1])})"
                return f"{self.generate_result(op.result)} ={self.generate_value(op.args[0])}->reshape({dims});"
            case "array.to_python":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->to_numpy();"
            case "array.fill":
                return iter_rec(op.args[0], len(op.args[0].type.shape), 0, [], lambda
                    indices: f"{self.generate_value(op.args[0])}->byIndices({{ {",".join(indices)} }}) = {self.generate_value(op.args[1])}(std::make_tuple({",".join(indices)}));")
            case "array.apply_scalar":
                res = f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->with_same_dims<{self.generate_type(op.result.type.element_type)}>();"
                res += iter_rec(op.args[0], len(op.args[0].type.shape), 0, [], lambda
                    indices: f"{self.generate_value(op.result)}->byIndices({{ {",".join(indices)} }}) = {self.generate_value(op.args[1])}({self.generate_value(op.args[0])}->byIndices({{ {",".join(indices)} }}));")
                return res
            case "array.binary_op":
                res = f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->with_same_dims<{self.generate_type(op.result.type.element_type)}>();"
                res += iter_rec(op.args[0], len(op.args[0].type.shape), 0, [], lambda
                    indices: f"{self.generate_value(op.result)}->byIndices({{ {",".join(indices)} }}) = {self.generate_value(op.args[2])}({self.generate_value(op.args[0])}->byIndices({{ {",".join(indices)} }}),{self.generate_value(op.args[1])}->byIndices({{ {",".join(indices)} }}));")
                return res
            case "array.compute":
                res = f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->with_same_dims<{self.generate_type(op.result.type.element_type)}>();"
                res += iter_rec(op.args[0], len(op.args[0].type.shape), 0, [], lambda
                    indices: f"{self.generate_value(op.result)}->byIndices({{ {",".join(indices)} }}) = {call(op.args[-1], ", ".join([f"{self.generate_value(a)}->byIndices({{ {",".join(indices)} }})" for a in op.args[:-1]]))};")
                return res
            case "array.get":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->byIndices({{ {",".join(map(self.generate_value, op.args[1:]))} }});"
            case "array.set":
                return f"{self.generate_value(op.args[0])}->byIndices({{ {",".join(map(self.generate_value, op.args[1:-1]))} }}) = {self.generate_value(op.args[-1])};"
            case "array.copy":
                return iter_rec(op.args[0], len(op.args[0].type.shape), 0, [], lambda
                    indices: f"{self.generate_value(op.args[0])}->byIndices({{ {",".join(indices)} }}) = {self.generate_value(op.args[1])}->byIndices({{ {",".join(indices)} }});")
            case "array.create_view":
                array = op.args[0]
                info_vector = f"view_infos{self.generate_unique_id()}"
                offset = f"offset{self.generate_unique_id()}"
                dims = f"dims{self.generate_unique_id()}"
                strides = f"strides{self.generate_unique_id()}"

                def create_view_info(arg):
                    match arg.type:
                        case ir.IntType():
                            return f"{{ .selective=true, .offset=(size_t){self.generate_value(arg)}, .end=0, .step=1 }}"
                        case ir.RecordType():
                            return f"{{ .selective=false, .offset=(size_t)std::get<0>({self.generate_value(arg)}), .end=(size_t)std::get<1>({self.generate_value(arg)}), .step=(size_t)std::get<2>({self.generate_value(arg)}) }}"
                        case _:
                            raise NotImplementedError(f"Type {arg.type} not implemented")

                return f"""
                std::vector<builtin::view_info> {info_vector};
                {"\n".join([f"{info_vector}.push_back({create_view_info(view_info)});" for view_info in op.args[1:]])}
                auto [{offset}, {dims}, {strides}] = builtin::computeOffsetAndStrides<{len(op.result.type.shape)}>({info_vector}, {self.generate_value(array)}->getStrides(), {self.generate_value(array)}->getOffset());
                {self.generate_result(op.result)} = std::make_shared<builtin::ndarray<{self.generate_type(array.type.element_type)},{len(op.result.type.shape)}>>({self.generate_value(array)}->get_data(), {dims}, {strides}, {offset});
                """
            case "array.dim":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->getDimension({self.generate_value(op.args[1])});"
            case "array.from_nested_list":
                num_dimensions = len(op.result.type.shape)
                dims = f"dims{self.generate_unique_id()}"
                res = f"std::array<int64_t,{num_dimensions}> {dims} = {{ {",".join(["-1" for i in range(num_dimensions)])} }};"
                res += f"builtin::dimensions_from_nested_list({self.generate_value(op.args[0])},0, {dims});"
                res += f"{self.generate_result(op.result)} = std::make_shared<builtin::ndarray<{self.generate_type(op.result.type.element_type)},{num_dimensions}>>({dims});\n"
                res += iter_rec(op.result, len(op.result.type.shape), 0, [], lambda
                    indices: f"{self.generate_value(op.result)}->byIndices({{ {",".join(indices)} }}) = {self.generate_value(op.args[1])}({self.generate_value(op.args[0])}->operator[]({")->operator[](".join(indices)}));")
                return res

            case "column.from_list":
                builder_type = get_column_builder(op.result.type.element_type)
                builder_var = f"col_builder{self.generate_unique_id()}"
                iter_var = f"v{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                for (const auto& {iter_var}: *{self.generate_value(op.args[0])}){{
                    {builder_var}.append({iter_var});
                }}
                {self.generate_result(op.result)} = {builder_var}.build();
                """
            case "column.to_python":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->to_python();"
            case "table.from_dict":
                list_var = f"cols{self.generate_unique_id()}"
                field_names = [field for field, _ in op.args[0].type.members]
                return f"""
    std::vector<std::pair<std::string,std::shared_ptr<builtin::tabular::Column>>> {list_var};
    {"\n".join([f"{list_var}.push_back({{ \"{field}\",std::get<{offset}>({self.generate_value(op.args[0])}) }});" for offset, field in enumerate(field_names)])}
    {self.generate_result(op.result)} = builtin::tabular::Table::from_columns({list_var});
                """
            case "table.to_python":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->to_python();"
            case "column.apply_scalar":
                builder_type = get_column_builder(op.result.type.element_type)
                builder_var = f"col_builder{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                {self.generate_value(op.args[0])}->iterate<{get_column_accessor(op.args[0].type.element_type)}>([&](auto iter_val){{
                    {builder_var}.append({self.generate_value(op.args[1])}(iter_val));
                }});
                {self.generate_result(op.result)} = {builder_var}.build();
                """
            case "column.binary_op":
                builder_type = get_column_builder(op.result.type.element_type)
                builder_var = f"col_builder{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                {self.generate_value(op.args[0])}->iterateZipped<{get_column_accessor(op.args[0].type.element_type)},{get_column_accessor(op.args[1].type.element_type)}>({self.generate_value(op.args[1])},[&](auto left_val, auto right_val){{
                    {builder_var}.append({self.generate_value(op.args[2])}(left_val, right_val));
                }});
                {self.generate_result(op.result)} = {builder_var}.build();
                """
            case "column.filter":
                builder_type = get_column_builder(op.result.type.element_type)
                builder_var = f"col_builder{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                {self.generate_value(op.args[0])}->iterateZipped<{get_column_accessor(op.args[0].type.element_type)},{get_column_accessor(op.args[1].type.element_type)}>({self.generate_value(op.args[1])},[&](auto left_val, auto right_val){{
                    if (right_val){{
                        {builder_var}.append(left_val);
                    }}
                }});
                {self.generate_result(op.result)} = {builder_var}.build();
                """
            case "column.aggregate":
                return f"""
    {self.generate_result(op.result)} = {self.generate_value(op.args[1])};
    {self.generate_value(op.args[0])}->iterate<{get_column_accessor(op.args[0].type.element_type)}>([&](auto iter_val){{
        {self.generate_value(op.result)}={self.generate_value(op.args[2])}({self.generate_value(op.result)},iter_val);
    }});
                 """
            case "column.unique":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->unique();"
            case "column.isin_column":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->isIn({self.generate_value(op.args[1])});"
            case "column.sequential":
                builder_type = get_column_builder(op.result.type.element_type)
                builder_var = f"col_builder{self.generate_unique_id()}"
                iter_var = f"v{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                for (int64_t {iter_var}=0;{iter_var}<{self.generate_value(op.args[0])};{iter_var}++){{
                    {builder_var}.append({iter_var});
                }}
                {self.generate_result(op.result)} = {builder_var}.build();
                """
            case "table.sort":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->sort({self.generate_value(op.args[1])},{self.generate_value(op.args[2])});"
            case "table.length":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->size();"
            case "table.slice":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->slice({self.generate_value(op.args[1])},{self.generate_value(op.args[2])});"
            case "table.get_column":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->getColumn(\"{self.escape_for_cpp(op.attributes["column"])}\");"
            case "table.select":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->selectColumns({{ {",".join([f"\"{self.escape_for_cpp(name)}\"" for name in op.attributes["columns"]])} }});"
            case "table.set_column":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->setColumn(\"{self.escape_for_cpp(op.attributes["column"])}\",{self.generate_value(op.args[1])});"
            case "table.add_index_column":
                builder_type = get_column_builder(ir.i64)
                builder_var = f"col_builder{self.generate_unique_id()}"
                iter_var = f"v{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                for (int64_t {iter_var}=0;{iter_var}<{self.generate_value(op.args[0])}->size();{iter_var}++){{
                    {builder_var}.append({iter_var});
                }}
                
                {self.generate_result(op.result)} = {self.generate_value(op.args[0])}->setColumn(\"{self.escape_for_cpp(op.attributes["name"])}\",{builder_var}.build());"""
            case "table.apply_row_wise_scalar":
                builder_type = get_column_builder(op.result.type.element_type)
                builder_var = f"col_builder{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                {self.generate_value(op.args[0])}->iterateBatches([&](auto batch){{
                {"\n".join([f"auto col_accessor{offset} = {get_column_accessor(m[1])}(batch->column({offset}));" for offset, m in enumerate(op.args[0].type.members)])}
                for (int64_t i = 0; i < batch->num_rows(); i++){{
                    {builder_var}.append({call(op.args[1], f"std::make_tuple({", ".join([f"col_accessor{offset}.access(i)" for offset, _ in enumerate(op.args[0].type.members)])})")});
                    }}
                }});
                {self.generate_result(op.result)} = {builder_var}.build();
                """
            case "table.compute":
                required_cols=op.attributes["input"]
                computed_cols=op.attributes["output"]
                builder_vars = {col: f"col_builder{self.generate_unique_id()}" for col in computed_cols}
                column_vars = {col: f"built_column_{self.generate_unique_id()}" for col in computed_cols}
                accessor_vars = {col: f"col_accessor{self.generate_unique_id()}" for col in required_cols}
                res_col_types = {col: type for col, type in op.result.type.members}
                arg_col_types = {col: type for col, type in op.args[0].type.members}
                arg_col_offsets = {col: offset for offset, (col, _) in enumerate(op.args[0].type.members)}
                res_col_offsets = {col: offset for offset, (col, _) in enumerate(op.result.type.members)}
                sorted_computed_cols = sorted(computed_cols, key=lambda col: res_col_offsets[col])
                builder_inits = [f"{get_column_builder(res_col_types[col])} {builder_vars[col]};" for col in
                                 computed_cols]
                return f"""
                {"\n".join(builder_inits)}
                {self.generate_value(op.args[0])}->iterateBatches([&](auto batch){{
                {"\n".join([f"auto {accessor_vars[col]} = {get_column_accessor(arg_col_types[col])}(batch->column({arg_col_offsets[col]}));" for col in required_cols])}
                for (int64_t i = 0; i < batch->num_rows(); i++){{
                    auto tmp={call(op.args[1], ", ".join([f"{accessor_vars[col]}.access(i)" for col in required_cols]))};
                    {"\n".join([f"{builder_vars[col]}.append({f"std::get<{computed_cols.index(col)}>(tmp)" if len(computed_cols) > 1 else "tmp"});" for col in computed_cols])}
                    }}
                }});
                {"\n".join([f"auto {column_vars[col]} = {builder_vars[col]}.build();" for col in computed_cols])}
                {self.generate_result(op.result)} = {self.generate_value(op.args[0])}{"".join([f"->setColumn(\"{col}\",{column_vars[col]})" for col in sorted_computed_cols])};
                """
            case "table.opt_boundary":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])};"
            case "table.filter_by_func":
                required_cols = op.attributes["columns"]
                table_cols = [col for col, _ in op.result.type.members]
                builder_vars = {col: f"col_builder{self.generate_unique_id()}" for col in table_cols}
                column_vars = {col: f"built_column_{self.generate_unique_id()}" for col in table_cols}
                accessor_vars = {col: f"col_accessor{self.generate_unique_id()}" for col in table_cols}
                res_col_types = {col: type for col, type in op.result.type.members}
                arg_col_types = {col: type for col, type in op.args[0].type.members}
                arg_col_offsets = {col: offset for offset, (col, _) in enumerate(op.args[0].type.members)}
                res_col_offsets = {col: offset for offset, (col, _) in enumerate(op.result.type.members)}
                sorted_computed_cols = sorted(table_cols, key=lambda col: res_col_offsets[col])
                builder_inits = [f"{get_column_builder(res_col_types[col])} {builder_vars[col]};" for col in table_cols]
                return f"""
                {"\n".join(builder_inits)}
                {self.generate_value(op.args[0])}->iterateBatches([&](auto batch){{
                {"\n".join([f"auto {accessor_vars[col]} = {get_column_accessor(arg_col_types[col])}(batch->column({arg_col_offsets[col]}));" for col in table_cols])}
                for (int64_t i = 0; i < batch->num_rows(); i++){{
                    auto tmp={call(op.args[1], ", ".join([f"{accessor_vars[col]}.access(i)" for col in required_cols]))};
                    if(tmp){{
                        {"\n".join([f"{builder_vars[col]}.append({accessor_vars[col]}.access(i));" for col in table_cols])}
                    }}
                    }}
                }});
                {"\n".join([f"auto {column_vars[col]} = {builder_vars[col]}.build();" for col in table_cols])}
                {self.generate_result(op.result)} =builtin::tabular::Table::from_columns({{ {",".join([f"{{\"{col}\",{column_vars[col]} }}" for col in sorted_computed_cols])} }});
                """
            case "table.join_inner":
                left = op.args[0]
                right = op.args[1]
                left_on=op.attributes["left_on"]
                right_on=op.attributes["right_on"]
                left_keep=op.attributes["left_keep"]
                right_keep=op.attributes["right_keep"]
                left_cols=[col for col,_ in left.type.members]
                right_cols=[col for col,_ in right.type.members]
                left_col_offsets={col: offset for offset,(col, t) in enumerate(left.type.members)}
                right_col_offsets={col: offset for offset,(col,t) in enumerate(right.type.members)}
                left_col_types = {col: type for col, type in left.type.members}
                right_col_types = {col: type for col, type in right.type.members}
                left_key_types = [left_col_types[col] for col in left_on]
                right_key_types = [right_col_types[col] for col in right_on]
                left_val_types = [left_col_types[col] for col in left_keep]
                right_val_types = [right_col_types[col] for col in right_keep]
                left_accessor_vars = {col: f"col_accessor{self.generate_unique_id()}" for col in left_on+left_keep}
                right_accessor_vars = {col: f"col_accessor{self.generate_unique_id()}" for col in right_on+right_keep}
                res_table_cols = [col for col, _ in op.result.type.members]
                res_col_types={col: type for col, type in op.result.type.members}
                column_vars = {col: f"built_column_{self.generate_unique_id()}" for col in res_table_cols}

                builder_vars = {col: f"col_builder{self.generate_unique_id()}" for col in res_table_cols}
                builder_inits = [f"{get_column_builder(res_col_types[col])} {builder_vars[col]};" for col in res_table_cols]
                sorted_res_cols = [col for col,_ in op.result.type.members]
                map_var=f"right_map{self.generate_unique_id()}"

                def _load_val(col):
                    if col in left_on or col in left_keep:
                        return f"{left_accessor_vars[col]}.access(i)"
                    else:
                        return f"std::get<{right_keep.index(col)}>(it->second)"
                return f"""
                std::unordered_multimap<std::tuple<{", ".join(map(self.generate_type, right_key_types))}>, std::tuple<{", ".join(map(self.generate_type, right_val_types))}>> {map_var};
                {self.generate_value(right)}->iterateBatches([&](auto batch){{
                    {"\n".join([f"auto {right_accessor_vars[col]} = {get_column_accessor(right_col_types[col])}(batch->column({right_col_offsets[col]}));" for col in right_cols if col in right_on+right_keep])}
                    for (int64_t i = 0; i < batch->num_rows(); i++){{
                        {map_var}.insert({{ {{ {", ".join([f"{right_accessor_vars[col]}.access(i)" for col in right_on])} }}, {{ {", ".join([f"{right_accessor_vars[col]}.access(i)" for col in right_keep])} }} }});
                    }}
                }});
                {"\n".join(builder_inits)}
                {self.generate_value(left)}->iterateBatches([&](auto batch){{
                    {"\n".join([f"auto {left_accessor_vars[col]} = {get_column_accessor(left_col_types[col])}(batch->column({left_col_offsets[col]}));" for col in left_cols if col in left_on+left_keep])}
                    for (int64_t i = 0; i < batch->num_rows(); i++){{
                        auto range = {map_var}.equal_range({{ {", ".join([f"{left_accessor_vars[col]}.access(i)" for col in left_on])} }});
                        for (auto it = range.first; it != range.second; ++it){{
                       {"\n".join([f"{builder_vars[col]}.append({_load_val(col)});" for col in res_table_cols])}
                        }}
                    }}
                }});
                {"\n".join([f"auto {column_vars[col]} = {builder_vars[col]}.build();" for col in res_table_cols])}
                {self.generate_result(op.result)} =builtin::tabular::Table::from_columns({{ {",".join([f"{{\"{col}\",{column_vars[col]} }}" for col in sorted_res_cols])} }});
                """
            case "table.aggregate":
                table = op.args[0]
                group_by = list(op.attributes["group_by"])
                col_types = {col: type for col, type in table.type.members}
                key_types=[col_types[col] for col in group_by]
                input_cols = list(op.attributes["input"])
                init_fn=op.args[1]
                agg_fn=op.args[2]
                finalize_fn=op.args[3]
                agg_type=init_fn.type.res_type
                output_cols = op.attributes["output"]
                table_col_offsets = {col: offset for offset, (col, t) in enumerate(table.type.members)}
                accessor_vars = {col: f"col_accessor{self.generate_unique_id()}" for col in group_by+input_cols}

                res_col_offsets = {col: offset for offset, (col, t) in enumerate(op.result.type.members)}
                res_table_cols = [col for col, _ in op.result.type.members]
                res_col_types={col: type for col, type in op.result.type.members}
                column_vars = {col: f"built_column_{self.generate_unique_id()}" for col in res_table_cols}

                builder_vars = {col: f"col_builder{self.generate_unique_id()}" for col in res_table_cols}
                builder_inits = [f"{get_column_builder(res_col_types[col])} {builder_vars[col]};" for col in res_table_cols]
                sorted_res_cols = [col for col,_ in op.result.type.members]
                map_var=f"agg_map{self.generate_unique_id()}"

                return f"""
                std::unordered_map<std::tuple<{", ".join(map(self.generate_type, key_types))}>, {self.generate_type(agg_type)}> {map_var};
                {self.generate_value(table)}->iterateBatches([&](auto batch){{
                    {"\n".join([f"auto {accessor_vars[col]} = {get_column_accessor(col_types[col])}(batch->column({table_col_offsets[col]}));" for col in col_types if col in group_by+input_cols])}
                    for (int64_t i = 0; i < batch->num_rows(); i++){{
                        auto key=std::make_tuple({", ".join([f"{accessor_vars[col]}.access(i)" for col in group_by])});
                        if(!{map_var}.contains(key)){{
                            {map_var}.insert({{ key, {call(init_fn,"")} }});
                        }}
                        {map_var}[key]={call(agg_fn, f"{map_var}[key], std::make_tuple({", ".join([f"{accessor_vars[col]}.access(i)" for col in input_cols]) })")};
                    }}
                }});
                {"\n".join(builder_inits)}
                for (const auto& [key, val]: {map_var}){{
                    {"\n".join([f"{builder_vars[col]}.append(std::get<{i}>(key));" for i,col in enumerate(group_by)])}
                    auto final_val={call(finalize_fn,"val")};
                    {"\n".join([f"{builder_vars[col]}.append(std::get<{i}>(final_val));" for i, col in enumerate(output_cols)])}
                }}
                {"\n".join([f"auto {column_vars[col]} = {builder_vars[col]}.build();" for col in res_table_cols])}
                {self.generate_result(op.result)} =builtin::tabular::Table::from_columns({{ {",".join([f"{{\"{col}\",{column_vars[col]} }}" for col in sorted_res_cols])} }});

                """
            case "table.join_left":
                left = op.args[0]
                right = op.args[1]
                left_on=op.attributes["left_on"]
                right_on=op.attributes["right_on"]
                left_keep=op.attributes["left_keep"]
                right_keep=op.attributes["right_keep"]
                left_cols=[col for col,_ in left.type.members]
                right_cols=[col for col,_ in right.type.members]
                left_col_offsets={col: offset for offset,(col, t) in enumerate(left.type.members)}
                right_col_offsets={col: offset for offset,(col,t) in enumerate(right.type.members)}
                left_col_types = {col: type for col, type in left.type.members}
                right_col_types = {col: type for col, type in right.type.members}
                left_key_types = [left_col_types[col] for col in left_on]
                right_key_types = [right_col_types[col] for col in right_on]
                left_val_types = [left_col_types[col] for col in left_keep]
                right_val_types = [right_col_types[col] for col in right_keep]
                left_accessor_vars = {col: f"col_accessor{self.generate_unique_id()}" for col in left_on+left_keep}
                right_accessor_vars = {col: f"col_accessor{self.generate_unique_id()}" for col in right_on+right_keep}
                res_table_cols = [col for col, _ in op.result.type.members]
                res_col_types={col: type for col, type in op.result.type.members}
                column_vars = {col: f"built_column_{self.generate_unique_id()}" for col in res_table_cols}

                builder_vars = {col: f"col_builder{self.generate_unique_id()}" for col in res_table_cols}
                builder_inits = [f"{get_column_builder(res_col_types[col])} {builder_vars[col]};" for col in res_table_cols]
                sorted_res_cols = [col for col,_ in op.result.type.members]
                map_var = f"right_map{self.generate_unique_id()}"

                def _load_val(col):
                    if col in left_on or col in left_keep:
                        return f"{left_accessor_vars[col]}.access(i)"
                    else:
                        return f"std::get<{right_keep.index(col)}>(it->second)"
                return f"""
                std::unordered_multimap<std::tuple<{", ".join(map(self.generate_type, right_key_types))}>, std::tuple<{", ".join(map(self.generate_type, right_val_types))}>> {map_var};
                {self.generate_value(right)}->iterateBatches([&](auto batch){{
                    {"\n".join([f"auto {right_accessor_vars[col]} = {get_column_accessor(right_col_types[col])}(batch->column({right_col_offsets[col]}));" for col in right_cols if col in right_on+right_keep])}
                    for (int64_t i = 0; i < batch->num_rows(); i++){{
                        {map_var}.insert({{ {{ {", ".join([f"{right_accessor_vars[col]}.access(i)" for col in right_on])} }}, {{ {", ".join([f"{right_accessor_vars[col]}.access(i)" for col in right_keep])} }} }});
                    }}
                }});
                {"\n".join(builder_inits)}
                {self.generate_value(left)}->iterateBatches([&](auto batch){{
                    {"\n".join([f"auto {left_accessor_vars[col]} = {get_column_accessor(left_col_types[col])}(batch->column({left_col_offsets[col]}));" for col in left_cols if col in left_on+left_keep])}
                    for (int64_t i = 0; i < batch->num_rows(); i++){{
                        bool foundMatch=false;
                        auto range = {map_var}.equal_range({{ {", ".join([f"{left_accessor_vars[col]}.access(i)" for col in left_on])} }});
                        for (auto it = range.first; it != range.second; ++it){{
                       {"\n".join([f"{builder_vars[col]}.append({_load_val(col)});" for col in res_table_cols])}
                            foundMatch=true;
                        }}
                        if(!foundMatch){{
                        {"\n".join([f"{builder_vars[col]}.append({left_accessor_vars[col]}.access(i));" if col in left_keep else f"{builder_vars[col]}.append(NAN);" for col in res_table_cols])}
                        }}
                    }}
                }});
                {"\n".join([f"auto {column_vars[col]} = {builder_vars[col]}.build();" for col in res_table_cols])}
                {self.generate_result(op.result)} =builtin::tabular::Table::from_columns({{ {",".join([f"{{\"{col}\",{column_vars[col]} }}" for col in sorted_res_cols])} }});
                """

            case "table.filter":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->filterByColumn({self.generate_value(op.args[1])});"
            case "column.length":
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->size();"
            case "column.value_by_index":
                self.enable_arrow = True
                return f"{self.generate_result(op.result)} = {self.generate_value(op.args[0])}->byIndex<{get_column_accessor(op.args[0].type.element_type)}>({self.generate_value(op.args[1])});"
            case "column.from_array":
                builder_type = get_column_builder(op.result.type.element_type)
                builder_var = f"col_builder{self.generate_unique_id()}"
                i_var = f"i{self.generate_unique_id()}"
                return f"""
                {builder_type} {builder_var};
                for (int64_t {i_var} = 0; {i_var} < {self.generate_value(op.args[0])}->size(); {i_var}++){{
                    {builder_var}.append({self.generate_value(op.args[0])}->byOffset( {i_var} ));
                }}
                {self.generate_result(op.result)} = {builder_var}.build();
                
                """
            case "benchmark.load_arrow":
                self.enable_arrow = True
                return f"{self.generate_result(op.result)} = builtin::tabular::load({self.generate_value(op.args[0])});"
            case "benchmark.read_file_to_lines":
                return f"{self.generate_result(op.result)} = builtin::read_file_to_lines({self.generate_value(op.args[0])});"

            case "date.parse":
                return f"{self.generate_result(op.result)} = builtin::date::parseDateToNanoseconds({self.generate_value(op.args[0])});"
            case "date.get_hour":
                return f"{self.generate_result(op.result)} = builtin::date::extractHour({self.generate_value(op.args[0])});"
            case "date.get_year":
                return f"{self.generate_result(op.result)} = builtin::date::extractYear({self.generate_value(op.args[0])});"
            case "scalar.float.isnan":
                return f"{self.generate_result(op.result)} = std::isnan({self.generate_value(op.args[0])});"
            case _:
                raise NotImplementedError(f"builtin {op.name} not implemented")

    def generate_op(self, op):
        import hipy.cppbackend.cppir as cppir
        match op:
            case ir.Undef(result=r):
                return f"{self.generate_result(op.result)} = {{}};"
            case ir.Free(v=v):
                match v.type:
                    case ir.ArrayType():
                        return f"{self.generate_value(v)}.reset();"
                    case _:
                        return "// free not implemented for this type"
            case ir.Constant(result=r, v=v):
                str = f"{self.generate_result(r)} = {self.generate_constant(v)};"
                if isinstance(r.type, ir.StringType):
                    self.global_constants += str + "\n"
                    return ""
                else:
                    return str
            case ir.Return(values=[value]):
                return f"return {self.generate_value(value)};"
            case ir.CallBuiltin():
                return self.generate_builtin(op)
            case ir.MakeRecord(values=values, result=r):
                vals = [self.generate_value(values[field]) for field, _ in r.type.members]
                return f"{self.generate_result(r)} = std::make_tuple({', '.join(vals)});"
            case ir.RecordGet(record=record, member=field, result=r):
                field_names = [m[0] for m in record.type.members]
                pos = field_names.index(field)
                return f"{self.generate_result(r)} = std::get<{pos}>({self.generate_value(record)});"
            case ir.PyImport(name=name, result=r):
                self.global_py_constants[self.generate_value(r)] = f"py::module::import(\"{(name)}\")";
                return ""
            case ir.PyGetAttr(on=on, name=attr, result=r):
                if self.generate_value(on) in self.global_py_constants:
                    self.global_py_constants[self.generate_value(r)] = f"{self.generate_value(on)}.attr(\"{attr}\")"
                    return ""
                name_var = f"py_attr_str{self.generate_unique_id()}"
                self.global_py_constants[name_var] = f"py::str(\"{self.escape_for_cpp(attr)}\")";
                return f"{self.generate_result(r)} = py::reinterpret_steal<py::object>(PyObject_GetAttr({self.generate_value(on)}.ptr(), {name_var}.ptr()));"
                #return f"{self.generate_result(r)} = {self.generate_value(on)}.attr(\"{attr}\");"
            case ir.PySetAttr(on=on, name=attr, value=value):
                self.enable_python = True
                return f"{self.generate_value(on)}.attr(\"{attr}\") = {self.generate_value(value)};"
            case ir.PythonCall(callable=fn, args=args, kw_args=kw_args, result=r):
                #if len(kw_args) ==0 and len(args) != 0:
                #    return f"{self.generate_result(r)} = py::reinterpret_steal <py::object> (PyObject_CallFunctionObjArgs({self.generate_value(fn)}.ptr(), {",".join([f"{self.generate_value(v)}.ptr()" for v in args])},NULL));"
                kw_args_def = ""
                kw_args_use = ""
                kw_args_id = self.generate_unique_id()
                if len(kw_args) > 0:
                    kw_args_def = f"""
    py::kwargs kwargs{kw_args_id};
    {"\n".join([f'kwargs{kw_args_id}["{k}"] = {self.generate_value(v)};' for k, v in kw_args])}
                    """
                    kw_args_use = f"{"," if args else ""} **kwargs{kw_args_id}"
                return f"{kw_args_def}{self.generate_result(r)} = {self.generate_value(fn)}({', '.join(map(self.generate_value, args))}{kw_args_use});"
            case ir.FunctionRef(name=name, closure=closure, result=r):
                if closure is None:
                    return f"{self.generate_result(r)} = &{name};"
                else:
                    return f"{self.generate_result(r)} = builtin::bound_fn(&{name}, {self.generate_value(closure)});"

            case ir.IfElse(cond=cond, ifBody=ifBody, elseBody=elseBody, results=results):
                return f"""
    {"\n".join(map(lambda r: f"{self.generate_result(r)};", results))}
    
    if ({self.generate_value(cond)}){{
    {"\n".join(map(self.generate_op, ifBody.ops[:-1]))}
    {"\n".join(map(lambda r: f"{self.generate_value(r[0])} = {self.generate_value(r[1])};", zip(results, ifBody.ops[-1].values)))}
    }} else {{
    {"\n".join(map(self.generate_op, elseBody.ops[:-1]))}
    {"\n".join(map(lambda r: f"{self.generate_value(r[0])} = {self.generate_value(r[1])};", zip(results, elseBody.ops[-1].values)))}
    }}
    """
            case ir.Call(name=func, args=args, result=r):
                return f"{self.generate_result(r)} = {func}({', '.join(map(self.generate_value, args))});"

            case cppir.CppOp():
                return op.produce(self)

            case ir.CallIndirect(fnref=fnref, args=args, result=r):
                return f"{self.generate_result(r)} = ({self.generate_value(fnref)})({', '.join(map(self.generate_value, args))});"
            case _:
                raise NotImplementedError(f"Op {op} not implemented")

    def generate_function(self, func: ir.Function):
        template = env.from_string("""
    inline {{ret_type}} {{fn_name}}({{fn_args}}){
    {{fn_ops}}
    }
    """)

        def generate_args():
            return ", ".join(map(lambda arg: f"{self.generate_type(arg.type)} val_{arg.id}", func.args))

        return template.render(ret_type=self.generate_type(func.res_type), fn_name=func.name, fn_args=generate_args(),
                               fn_ops=textwrap.indent("\n".join(map(lambda op: self.generate_op(op), func.body.ops)),
                                                      " " * 4))

    def generate_function_decl(self, func: ir.Function):
        template = env.from_string("""
    inline {{ret_type}} {{fn_name}}({{fn_args}});
    """)

        def generate_args():
            return ", ".join(map(lambda arg: f"{self.generate_type(arg.type)} val_{arg.id}", func.args))

        return template.render(ret_type=self.generate_type(func.res_type), fn_name=func.name, fn_args=generate_args())

    def generate_module(self, module: ir.Module):
        res = ""
        for func in module.block.ops:
            res += self.generate_function_decl(func)
        for func in module.block.ops:
            res += self.generate_function(func)
        return res

    def run(self):
        python_init = ""
        for el in self.module.imports.items():
            python_init += f'mainModule.add_object("{el[0]}",py::module_::import("{el[1]}"));\n'
        for name, el in self.module.py_functions.items():
            python_init += f'py::eval<pybind11::eval_single_statement>("{self.escape_for_cpp(el)}");\n'
        method_definitions = self.generate_module(self.module)
        self.enable_python |= self.enable_arrow
        self.enable_python |= self.enable_numpy
        return {"method_definitions": method_definitions,
                "fn": self.fn_name,
                "init_python": python_init,
                "py_enabled": "1" if self.enable_python else "0",
                "arrow_enabled": "1" if self.enable_arrow else "0",
                "numpy_enabled": "1" if self.enable_numpy else "0",
                "global_constants": self.global_constants,
                "global_py_decl": "\n".join([f"py::object {name};" for name in self.global_py_constants.keys()]),
                "global_py_init": "\n".join([f"{name} = {value};" for name, value in self.global_py_constants.items()]),
                "global_py_deinit": "\n".join([f"{name} = {{}};" for name, value in self.global_py_constants.items()])}


def run(fn_name, module: ir.Module, release=False):
    params = CPPBackend(fn_name, module, release).run()
    def_str = env.get_template('standalone.cpp').render(**params)
    return write_compile_run_cpp(def_str, release=release)


def run_udf_scan(fn_name, module: ir.Module,columns: Dict[str, Any],data_file, release=True):
    res_type=module.func(fn_name).res_type

    params = CPPBackend(fn_name, module, release).run()
    params["arrow_enabled"]="1"
    params["py_enabled"]="1"
    params["numpy_enabled"]="1"
    params["column_accessors"]= "\n".join([f"auto col_{col} = {get_column_accessor(columns[col])}(batch->column({i}));" for i,col in enumerate(columns)])
    params["column_vals"]= ",".join([f"col_{col}.access(i)" for col in columns])
    def_str = env.get_template('udf_eval.cpp').render(**params,data_file=data_file,res_builder_type=get_column_builder(res_type))
    return write_compile_run_cpp(def_str, release=release)
