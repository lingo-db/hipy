__HIPY_MODULE__ = "sklearn.pipeline"
import numpy as np
import pandas as pd
import hipy.lib.pandas

import hipy.lib.numpy
import sys
from hipy.value import static_object, HLCFunctionValue
import hipy
from hipy import intrinsics, ir
from hipy.value import SimpleType, Value, Type, raw_module
import hipy.lib.numpy as _np
import sklearn.pipeline as _pipeline
original=raw_module(_pipeline)
hipy.register(sys.modules[__name__])
@hipy.classdef
class Pipeline(static_object["steps",]):
    def __init__(self, steps):
        super().__init__(lambda args: Pipeline(*args), steps)

    @hipy.compiled_function
    def __topython__(self):
        obj= original.Pipeline(self.steps)
        return obj

    @staticmethod
    def __from_constant__(value:_pipeline.Pipeline,context):
        steps=[]
        for i in value.steps:
            name=i[0]
            step=i[1]
            corresponding=context.get_corresponding_class(type(step))
            steps.append(context.create_tuple([context.constant(name), corresponding.__from_constant__(step,context)]))
        return context.wrap(Pipeline(context.create_list(steps)))

    @hipy.compiled_function
    def predict(self, elts):
        data=elts
        for i in range(0,len(self.steps)-1):
            name=self.steps[i][0]
            step=self.steps[i][1]
            data=step.transform(data)
        return self.steps[-1][1].predict(data)
