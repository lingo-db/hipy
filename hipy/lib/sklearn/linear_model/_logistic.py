__HIPY_MODULE__ = "sklearn.linear_model._logistic"
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
import sklearn.linear_model._logistic as _logistic
original=raw_module(_logistic)
hipy.register(sys.modules[__name__])

@hipy.classdef
class LogisticRegression(static_object["penalty", "dual","tol","C","fit_intercept","intercept_scaling","class_weight","random_state","solver","max_iter","multi_class","verbose","warm_start","n_jobs","l1_ratio","classes_","coef_","intercept_","n_features_in_","feature_names_in_","n_iter_"]):
    def __init__(self, penalty, dual,tol,C,fit_intercept,intercept_scaling,class_weight,random_state,solver,max_iter,multi_class,verbose,warm_start,n_jobs,l1_ratio,classes_,coef_,intercept_,n_features_in_,feature_names_in_,n_iter_):
        super().__init__(lambda args: LogisticRegression(*args), penalty, dual,tol,C,fit_intercept,intercept_scaling,class_weight,random_state,solver,max_iter,multi_class,verbose,warm_start,n_jobs,l1_ratio,classes_,coef_,intercept_,n_features_in_,feature_names_in_,n_iter_)

    @hipy.compiled_function
    def __topython__(self):
        obj= original.LogisticRegression()
        obj.penalty=self.penalty
        obj.dual=self.dual
        obj.tol=self.tol
        obj.C=self.C
        obj.fit_intercept=self.fit_intercept
        obj.intercept_scaling=self.intercept_scaling
        obj.class_weight=self.class_weight
        obj.random_state=self.random_state
        obj.solver=self.solver
        obj.max_iter=self.max_iter
        obj.multi_class=self.multi_class
        obj.verbose=self.verbose
        obj.warm_start=self.warm_start
        obj.n_jobs=self.n_jobs
        obj.l1_ratio=self.l1_ratio
        obj.classes_=self.classes_
        obj.coef_=self.coef_
        obj.intercept_=self.intercept_
        obj.n_features_in_=self.n_features_in_
        #obj.feature_names_in_=self.feature_names_in_
        obj.n_iter_=self.n_iter_
        return obj

    @staticmethod
    def __from_constant__(value:_logistic.LogisticRegression,context):
        penalty=context.constant(value.penalty)
        dual=context.constant(value.dual)
        tol=context.constant(value.tol)
        C=context.constant(value.C)
        fit_intercept=context.constant(value.fit_intercept)
        intercept_scaling=context.constant(value.intercept_scaling)
        class_weight=context.constant(value.class_weight)
        random_state=context.constant(value.random_state)
        solver=context.constant(value.solver)
        max_iter=context.constant(value.max_iter)
        multi_class=context.constant(value.multi_class)
        verbose=context.constant(value.verbose)
        warm_start=context.constant(value.warm_start)
        n_jobs=context.constant(value.n_jobs)
        l1_ratio=context.constant(value.l1_ratio)
        classes_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.wrap(_np._const_float64(v)) for v in value.classes_.tolist()])])
        coef_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.create_list([context.wrap(_np._const_float64(v2)) for v2 in v]) for v in value.coef_.tolist()])])
        intercept_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.wrap(_np._const_float64(v)) for v in value.intercept_.tolist()])])
        n_features_in_=context.constant(value.n_features_in_)
        n_iter_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.wrap(_np._const_float64(v)) for v in value.n_iter_.tolist()])])
        feature_names_in_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.constant(v) for v in value.feature_names_in_.tolist()])])
        return context.wrap(LogisticRegression(penalty, dual,tol,C,fit_intercept,intercept_scaling,class_weight,random_state,solver,max_iter,multi_class,verbose,warm_start,n_jobs,l1_ratio,classes_,coef_,intercept_,n_features_in_,feature_names_in_,n_iter_))

    @hipy.compiled_function
    def predict(self, elts):

        def apply_to_single(e, coefs, intercept):
            s = intercept.shape[0]
            curr =[]
            for j in range(0, s):
                r=intercept[j]
                for i in range(0, len(e)):
                    r=r+(e[i]*coefs[j,i])
                curr.append(r)
            if s == 1:
                return int(curr[0] > 0)
            else:
                argmax= 0
                max= curr[0]
                for i in range(1, s):
                    if curr[i] > max:
                        max= curr[i]
                        argmax= i
                return argmax
        feature_names=self.feature_names_in_
        if intrinsics.isa(elts, list):
            res=[]
            for e in elts:
                res.append(apply_to_single(e, self.coef_, self.intercept_))
            return res
        elif intrinsics.isa(elts, pd.DataFrame):
            column_names=intrinsics.as_constant([feature_names[i] for i in range(0, feature_names.shape[0])])
            coefs = intrinsics.as_constant(self.coef_)
            intercept = intrinsics.as_constant(self.intercept_)
            #todo: convert this back to numpy array (conversions from/to series can then be optimized away later)
            return elts.apply(lambda e: apply_to_single([e[c] for c in column_names],coefs,intercept), axis=1)
        else:
            intrinsics.not_implemented()