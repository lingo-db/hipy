__HIPY_MODULE__ = "sklearn.cluster._kmeans"
import numpy as np
import pandas as pd

import hipy.lib.numpy
import sys
from hipy.value import static_object, HLCFunctionValue
import hipy
from hipy import intrinsics, ir
from hipy.value import SimpleType, Value, Type, raw_module
import hipy.lib.numpy as _np
import sklearn.cluster._kmeans as _kmeans
original=raw_module(_kmeans)
hipy.register(sys.modules[__name__])
@hipy.classdef
class KMeans(static_object["n_clusters", "cluster_centers_"]):
    def __init__(self, n_clusters,cluster_centers_):
        super().__init__(lambda args: KMeans(*args), n_clusters,cluster_centers_)

    @hipy.compiled_function
    def __topython__(self):
        obj= original.KMeans()
        obj.n_clusters=self.n_clusters
        obj.cluster_centers_=self.cluster_centers_
        return obj

    @staticmethod
    def __from_constant__(value:_kmeans.KMeans,context):
        n_clusters=context.constant(value.n_clusters)
        cluster_centers_=context.perform_call(context.wrap(HLCFunctionValue(_np.array)),[context.create_list([context.create_list([context.wrap(_np._const_float64(v2)) for v2 in v]) for v in value.cluster_centers_.tolist()])])
        return context.wrap(KMeans(n_clusters,cluster_centers_))

    @hipy.compiled_function
    def predict(self, elts):
        def euclidean_distance(point1, point2,x):
            res = 0
            for i in range(0, len(point1)):
                res = res + ((point1[i] - point2[x,i]) ** 2)
            return np.sqrt(res)
        def closest_point(point, points):
            s=points.shape[0]
            res = 0
            res_dist = euclidean_distance(point, points,0)
            for i in range(1, s):
                d = euclidean_distance(point, points,i)
                if d < res_dist:
                    res = i
                    res_dist = d
            return res
        if intrinsics.isa(elts, pd.DataFrame):
            points=intrinsics.as_constant(self.cluster_centers_)
            columns=intrinsics.as_constant(elts._columns())
            return elts.apply(lambda row: closest_point([row[c] for c in columns],points), axis=1)
        else:
            intrinsics.not_implemented()


