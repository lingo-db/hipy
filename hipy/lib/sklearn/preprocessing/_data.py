__HIPY_MODULE__ = "sklearn.preprocessing._data"
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
import sklearn.preprocessing._data as _data
original=raw_module(_data)
hipy.register(sys.modules[__name__])
@hipy.classdef
class MinMaxScaler(static_object["scale_","data_min_","feature_names_in_"]):
    def __init__(self, scale_,data_min_,feature_names_in_):
        super().__init__(lambda args: MinMaxScaler(*args), scale_,data_min_,feature_names_in_)

    @hipy.compiled_function
    def __topython__(self):
        obj= original.MinMaxScaler()
        obj.scale_=self.scale_
        obj.min_=self.min_
        return obj

    @staticmethod
    def __from_constant__(value:_data.MinMaxScaler,context):
        scale_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.wrap(_np._const_float64(v)) for v in value.scale_.tolist()])])
        data_min_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.wrap(_np._const_float64(v)) for v in value.data_min_.tolist()])])
        feature_names_in_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.constant(v) for v in value.feature_names_in_.tolist()])])
        return context.wrap(MinMaxScaler(scale_,data_min_,feature_names_in_))

    @hipy.compiled_function
    def transform(self, elts):
        if intrinsics.isa(elts, pd.DataFrame):
            n_features = self.feature_names_in_.shape[0]
            res=elts._clone()
            for i in range(0,n_features):
                res[self.feature_names_in_[i]]=((res[self.feature_names_in_[i]]-self.data_min_[i])*self.scale_[i])
            return res
        else:
            intrinsics.not_implemented()

