#define PY_ENABLED {{py_enabled}}
#define ARROW_ENABLED {{arrow_enabled}}
#define NUMPY_ENABLED {{numpy_enabled}}
#include "builtin.h"
#if ARROW_ENABLED==1
#include "builtin_arrow.h"
#endif
#if NUMPY_ENABLED==1
#include "builtin_numpy.h"
#endif
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <sstream>
#if PY_ENABLED==1
#include<pybind11/embed.h>
namespace py = pybind11;
#endif
{{global_constants}}
{{global_py_decl}}

{{method_definitions}}
int main(){
    #if PY_ENABLED==1
    py::scoped_interpreter guard{};
    auto mainModule=py::module_::import("__main__");
    {{init_python}}
    #endif
    #if ARROW_ENABLED==1
    arrow::py::import_pyarrow();
    #endif
    {{global_py_init}}
    {{fn}}();
    {{global_py_deinit}}

    return 0;
}