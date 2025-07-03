import inspect


def compiled_function(func):
    from hipy.function import HLCFunction
    return HLCFunction(func)


def raw(fn):
    from hipy.function import HLCFunction
    if "_context" not in inspect.signature(fn).parameters:
        def with_context(*args, **kwargs):
            if "_context" in kwargs:
                del kwargs["_context"]
            return fn(*args, **kwargs)

        return HLCFunction(fn, with_context)

    return HLCFunction(fn, fn)


def classdef(cls):
    from hipy.function import HLCFunction, HLCMethod
    to_method_wrap = []
    for name, value in cls.__dict__.items():
        if isinstance(value, HLCFunction):
            to_method_wrap.append(name)
    extra_dict = {}
    for name in to_method_wrap:
        extra_dict[name] = cls.__dict__[name]
        setattr(cls, name, lambda self, *args, **kwargs: getattr(self, name)(*args, **kwargs))

    old_init = cls.__init__ if hasattr(cls, "__init__") else None
    def __init__(self, *args, **kwargs):
        if old_init is not None:
            old_init(self, *args, **kwargs)
        for name in to_method_wrap:
            if hasattr(self, name):
                setattr(self, "_super_" + name, getattr(self, name))
            setattr(self, name, HLCMethod(extra_dict[name], self))
    cls.__init__ = __init__
    cls.__hipy__ = True
    return cls


def raw_generator(fn):
    from hipy.function import GeneratorFunction
    return GeneratorFunction(fn)