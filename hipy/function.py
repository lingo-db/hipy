import inspect
import sys


class HLCFunction:
    def __init__(self, pyfunc, compiled_fn=None):
        self.pyfunc = pyfunc
        self.compiled_fn = compiled_fn

    def get_compiled_fn(self):
        from hipy.compiler import stage_and_compile
        if self.compiled_fn is None:
            self.compiled_fn = stage_and_compile(self.pyfunc)
        return self.compiled_fn

    def __call__(self, *args, **kwargs):
        return self.pyfunc(*args, **kwargs)

    def get_name(self):
        return self.pyfunc.__name__


class HLCMethod:
    def __init__(self, func: HLCFunction, self_value):
        self.func = func
        self.self_value = self_value

    def __call__(self, *args, **kwargs):
        return self.func(self.self_value, *args, **kwargs)


class GeneratorFunction:
    def __init__(self, pyfunc):
        self.pyfunc = pyfunc

    def get_compiled_fn(self):
        from hipy.compiler import stage_and_compile
        if self.compiled_fn is None:
            self.compiled_fn = stage_and_compile(self.pyfunc)
        return self.compiled_fn

    def __call__(self, *args, **kwargs):
        return self.pyfunc(*args, **kwargs)
