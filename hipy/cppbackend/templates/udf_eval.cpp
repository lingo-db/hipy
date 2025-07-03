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
#include <chrono>
#include "json.h"
#if PY_ENABLED==1
#include<pybind11/embed.h>
namespace py = pybind11;
using json = nlohmann::json;

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
    {{res_builder_type}} res_builder;
    auto table=builtin::tabular::load("{{data_file}}");
    auto start = std::chrono::high_resolution_clock::now();
    table->iterateBatches([&](auto batch){
        {{column_accessors}}
        for (int i=0;i<batch->num_rows();i++){
            res_builder.append({{fn}}({{column_vals}}));
        }
    });
        auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;

        json output;
    output["runtime"] = duration.count(); // Duration in seconds
    output["res"] = res_builder.build()->getData()->ToString();
    // Writing JSON to stdout
    std::cout << output.dump(4) << std::endl; // Dump with indentation of 4 spaces
    {{global_py_deinit}}

    return 0;
}