from hipy.decorators import compiled_function, raw, classdef, raw_generator

mocked_modules = {}

def register(module):
    global mocked_modules
    mocked_modules[module.__HIPY_MODULE__] = module

class global_const:
    def __init__(self, value):
        self.value = value