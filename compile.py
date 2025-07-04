import importlib.util
import json
import sys
import shutil

import hipy.compiler
import hipy.mlirbackend
import hipy.lib.builtins
import hipy.ir as ir
from hipy.value import SimpleType

file = sys.argv[1] if len(sys.argv) > 1 else None
function = sys.argv[2] if len(sys.argv) > 2 else None

arg_types_json = json.loads(sys.argv[3]) if len(sys.argv) > 3 else []



if file is None or function is None or not arg_types_json:
    print("Usage: python compile.py <file> <function> <arg_types_json> [<output_file>]")
    print("Example: python compile.py test.py extractType '[\"str\"]'")
    sys.exit(1)


arg_types=[]
for arg_type in arg_types_json:
    if arg_type == 'str':
        arg_types.append(SimpleType(hipy.lib.builtins.str, ir.string))
    elif arg_type == 'int':
        arg_types.append(SimpleType(hipy.lib.builtins.int, ir.int))
    elif arg_type == 'float':
        arg_types.append(SimpleType(hipy.lib.builtins.float, ir.f64))
    else:
        raise ValueError(f"Unknown argument type: {arg_type}")


def load_module_from_file(module_name: str, file_path: str):
    # handle case where file path does not end with .py: copy file to same directory with .py extension
    old_file_path = file_path
    if not file_path.endswith('.py'):
        file_path += '.py'
        # perform the copy the file to the new path
        shutil.copy(old_file_path, file_path)


    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module_from_file("udf_module", file)
func = getattr(mod, function)

module = hipy.compiler.compile(func, arg_types=arg_types, fallback=False,
                               debug=False)
mlir = hipy.mlirbackend.compile(module)
# write mlir to output file if provided, otherwise print to stdout
output_file = sys.argv[4] if len(sys.argv) > 4 else None
if output_file:
    with open(output_file, "w") as program_file:
        program_file.write(str(mlir))
else:
    print(str(mlir))

