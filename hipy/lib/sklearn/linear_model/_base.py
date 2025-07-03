__HIPY_MODULE__ = "sklearn.linear_model._base"
import numpy as np
import hipy.lib.numpy
import sys
from hipy.value import static_object, HLCFunctionValue
import hipy
from hipy import intrinsics, ir
from hipy.value import SimpleType, Value, Type, raw_module
import hipy.lib.numpy as _np
import sklearn.linear_model._base as _base
original=raw_module(_base)
hipy.register(sys.modules[__name__])
@hipy.classdef
class LinearRegression(static_object["fit_intercept", "copy_X","n_jobs","positive","coef_","rank_","singular_","intercept_","n_features_in_"]):
    def __init__(self, fit_intercept, copy_X,n_jobs,positive,coef_,rank_,singular_,intercept_,n_features_in_):
        super().__init__(lambda args: slice(*args), fit_intercept, copy_X,n_jobs,positive,coef_,rank_,singular_,intercept_,n_features_in_)

    @hipy.compiled_function
    def __topython__(self):
        obj= original.LinearRegression()
        obj.fit_intercept=self.fit_intercept
        obj.copy_X=self.copy_X
        obj.n_jobs=self.n_jobs
        obj.positive=self.positive
        obj.coef_=self.coef_
        obj.rank_=self.rank_
        obj.singular_=self.singular_
        obj.intercept_=self.intercept_
        obj.n_features_in_=self.n_features_in_
        return obj

    @staticmethod
    def __from_constant__(value:_base.LinearRegression,context):
        fit_intercept=context.constant(value.fit_intercept)
        copy_X=context.constant(value.copy_X)
        n_jobs=context.constant(value.n_jobs)
        positive=context.constant(value.positive)
        coef_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.wrap(_np._const_float64(v)) for v in value.coef_.tolist()])])
        rank_= context.constant(value.rank_)
        singular_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.wrap(_np._const_float64(v)) for v in value.coef_.tolist()])])
        intercept_=context.wrap(_np._const_float64(value.intercept_))
        n_features_in_=context.constant(value.n_features_in_)
        if hasattr(value,"feature_names_in_"):
            raise NotImplementedError()
        return context.wrap(LinearRegression(fit_intercept, copy_X,n_jobs,positive,coef_,rank_,singular_,intercept_,n_features_in_))

    @hipy.compiled_function
    def predict(self, elts):
        if intrinsics.isa(elts, list):
            res=[]
            for e in elts:
                r=self.intercept_
                for i in range(0, len(e)):
                    r=r+(e[i]*self.coef_[i])
                res.append(r)
            return res
        else:
            intrinsics.not_implemented()
