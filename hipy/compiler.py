import abc
import ast
import builtins
import copy
import inspect
import sys
import textwrap
import time

from hipy import ir
from hipy.function import HLCFunction
from hipy.context import Context, ValueAlias, NestedValueAlias, ConvertedToPython, ValUsage

import networkx as nx


class StageContext:
    def __init__(self, block, globals, nested, current_functions):
        self.block = block
        self.globals = globals
        self.available_variables = set()
        self.nested = nested
        self.action_id = 0
        self.current_functions = current_functions


def stage_expr_list(expr_list, context: StageContext, lineno, col_offset):
    res = []
    for expr in expr_list:
        res.append(stage_expr(expr, context))
    return ast.List(elts=res, ctx=ast.Load(), lineno=lineno, col_offset=col_offset)


def stage_context_call(name, kwargs, lineno, col_offset, context: StageContext):
    kwargs["_action_id"] = ast.Constant(value=context.action_id, lineno=lineno, col_offset=col_offset)
    context.action_id += 1
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="_context", ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                           attr=name, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
        keywords=[ast.keyword(arg=k, value=v, lineno=lineno, col_offset=col_offset) for k, v in
                  kwargs.items()],
        args=[], lineno=lineno, col_offset=col_offset)


def get_aug_method(op):
    match op:
        case ast.Add():
            return "__iadd__"
        case ast.Sub():
            return "__isub__"
        case ast.Mult():
            return "__imul__"
        case ast.Div():
            return "__itruediv__"
        case ast.Mod():
            return "__imod__"
        case ast.Pow():
            return "__ipow__"
        case ast.LShift():
            return "__ilshift__"
        case ast.RShift():
            return "__irshift__"
        case ast.BitOr():
            return "__ior__"
        case ast.BitXor():
            return "__ixor__"
        case ast.BitAnd():
            return "__iand__"
        case ast.FloorDiv():
            return "__ifloordiv__"
        case _:
            raise NotImplementedError()


def get_binop_methods(op):
    match op:
        case ast.Add():
            return "__add__", "__radd__"
        case ast.Sub():
            return "__sub__", "__rsub__"
        case ast.Mult():
            return "__mul__", "__rmul__"
        case ast.Div():
            return "__truediv__", "__rtruediv__"
        case ast.Mod():
            return "__mod__", "__rmod__"
        case ast.Pow():
            return "__pow__", "__rpow__"
        case ast.LShift():
            return "__lshift__", "__rlshift__"
        case ast.RShift():
            return "__rshift__", "__rrshift__"
        case ast.BitOr():
            return "__or__", "__ror__"
        case ast.BitXor():
            return "__xor__", "__rxor__"
        case ast.BitAnd():
            return "__and__", "__rand__"
        case ast.FloorDiv():
            return "__floordiv__", "__rfloordiv__"
        case ast.Eq():
            return "__eq__", "__eq__"
        case ast.NotEq():
            return "__ne__", "__ne__"
        case ast.Lt():
            return "__lt__", "__gt__"
        case ast.LtE():
            return "__le__", "__ge__"
        case ast.Gt():
            return "__gt__", "__lt__"
        case ast.GtE():
            return "__ge__", "__le__"

        case _:
            raise NotImplementedError()


tmp_name_counter = 0


def get_tmp_name():
    global tmp_name_counter
    tmp_name_counter += 1
    return f"__hipy_tmp_var_{tmp_name_counter}"


lambda_fn_counter = 0


class BindFreeVars(ast.NodeTransformer):
    def __init__(self, available_variables, not_bind, lineno, col_offset):
        self.ids = set()
        self.available_variables = available_variables
        self.not_bind = not_bind
        self.lineno = lineno
        self.col_offset = col_offset

    def visit_Name(self, node):
        if node.id in self.available_variables and node.id not in self.not_bind:
            self.ids.add(node.id)
            return ast.Subscript(
                value=ast.Name(id="__closure__", ctx=ast.Load(), lineno=self.lineno, col_offset=self.col_offset),
                slice=ast.Constant(value=node.id, lineno=self.lineno, col_offset=self.col_offset), lineno=self.lineno,
                col_offset=self.col_offset, ctx=ast.Load())
        else:
            return node


def stage_expr(expr, context: StageContext):
    match expr:
        case ast.Constant(value=val, lineno=lineno, col_offset=col_offset):
            return stage_context_call("constant",
                                      {"val": ast.Constant(value=val, lineno=lineno, col_offset=col_offset)}, lineno,
                                      col_offset, context)
        case ast.Attribute(value=value, attr=attr, lineno=lineno, col_offset=col_offset):
            return stage_context_call("get_attr",
                                      {"val": stage_expr(value, context),
                                       "attr": ast.Constant(value=attr, lineno=lineno, col_offset=col_offset)},
                                      lineno, col_offset, context)
        case ast.List(elts=elts, ctx=ctx, lineno=lineno, col_offset=col_offset):
            return stage_context_call("create_list",
                                      {"elts": stage_expr_list(elts, context, lineno, col_offset)}, lineno, col_offset,
                                      context)

        case ast.ListComp(elt=elt, generators=[ast.comprehension(target=ast.Name(id=targetname), iter=iter, ifs=ifs)],
                          lineno=lineno, col_offset=col_offset):
            def wrap_ifs(ifs, expr):
                if len(ifs) == 0:
                    return ast.Expr(value=expr, lineno=lineno, col_offset=col_offset)
                else:
                    return ast.If(test=ifs[0], body=[wrap_ifs(ifs[1:], expr)], orelse=[], lineno=lineno,
                                  col_offset=col_offset)

            list_var = get_tmp_name()
            assign_empty_list = ast.Assign(
                targets=[ast.Name(list_var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset)],
                value=ast.List(elts=[], ctx=ast.Load(), lineno=lineno, col_offset=col_offset), lineno=lineno,
                col_offset=col_offset)
            list_val = ast.Name(id=list_var, lineno=lineno, col_offset=col_offset, ctx=ast.Load())
            forStmt = ast.For(target=ast.Name(targetname, ctx=ast.Store(), lineno=lineno, col_offset=col_offset),
                              iter=iter,
                              body=[wrap_ifs(ifs, ast.Call(
                                  func=ast.Attribute(value=list_val, attr="append", lineno=lineno,
                                                     col_offset=col_offset,
                                                     ctx=ast.Load()),
                                  args=[elt], keywords=[], lineno=lineno, col_offset=col_offset))], orelse=[],
                              lineno=lineno, col_offset=col_offset)
            stage_stmt(assign_empty_list, context)
            stage_stmt(forStmt, context)
            return stage_context_call("get_by_name",
                                      {"val": list_val,
                                       "name": ast.Constant(list_var, lineno=lineno, col_offset=col_offset)},
                                      lineno, col_offset, context)
        case ast.DictComp(key=key_expr, value=value_expr,
                          generators=[ast.comprehension(target=ast.Name(id=targetname), iter=iter, ifs=ifs)],
                          lineno=lineno, col_offset=col_offset):
            def wrap_ifs(ifs, expr):
                if len(ifs) == 0:
                    return ast.Expr(value=expr, lineno=lineno, col_offset=col_offset)
                else:
                    return ast.If(test=ifs[0], body=[wrap_ifs(ifs[1:], expr)], orelse=[], lineno=lineno,
                                  col_offset=col_offset)

            dict_var = get_tmp_name()
            assign_empty_dict = ast.Assign(
                targets=[ast.Name(dict_var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset)],
                value=ast.Dict(keys=[], values=[], ctx=ast.Load(), lineno=lineno, col_offset=col_offset), lineno=lineno,
                col_offset=col_offset)
            dict_val = ast.Name(id=dict_var, lineno=lineno, col_offset=col_offset, ctx=ast.Load())
            forStmt = ast.For(target=ast.Name(targetname, ctx=ast.Store(), lineno=lineno, col_offset=col_offset),
                              iter=iter,
                              body=[wrap_ifs(ifs, ast.Call(
                                  func=ast.Attribute(value=dict_val, attr="__setitem__", lineno=lineno,
                                                     col_offset=col_offset,
                                                     ctx=ast.Load()),
                                  args=[key_expr, value_expr], keywords=[], lineno=lineno, col_offset=col_offset))],
                              orelse=[], lineno=lineno, col_offset=col_offset)
            stage_stmt(assign_empty_dict, context)
            stage_stmt(forStmt, context)
            return stage_context_call("get_by_name",
                                      {"val": dict_val,
                                       "name": ast.Constant(dict_var, lineno=lineno, col_offset=col_offset)},
                                      lineno, col_offset, context)

        case ast.Slice(lower=start, upper=stop, step=step, lineno=lineno, col_offset=col_offset):
            if start is None:
                start = ast.Constant(value=None, lineno=lineno, col_offset=col_offset)
            if stop is None:
                stop = ast.Constant(value=None, lineno=lineno, col_offset=col_offset)
            if step is None:
                step = ast.Constant(value=None, lineno=lineno, col_offset=col_offset)

            return stage_context_call("create_slice",
                                      {"start": stage_expr(start, context), "stop": stage_expr(stop, context),
                                       "step": stage_expr(step, context)}, lineno, col_offset, context)

        case ast.Dict(keys=keys, values=values, lineno=lineno, col_offset=col_offset):
            return stage_context_call("create_dict",
                                      {"keys": stage_expr_list(keys, context, lineno, col_offset),
                                       "values": stage_expr_list(values, context, lineno, col_offset)}, lineno,
                                      col_offset, context)
        case ast.Tuple(elts=elts, ctx=ctx, lineno=lineno, col_offset=col_offset):
            return stage_context_call("create_tuple",
                                      {"elts": stage_expr_list(elts, context, lineno, col_offset)}, lineno, col_offset,
                                      context)
        case ast.Subscript(value=value, slice=slice, ctx=ctx, lineno=lineno, col_offset=col_offset):
            return stage_context_call("get_item",
                                      {"val": stage_expr(value, context), "item": stage_expr(slice, context)},
                                      lineno, col_offset, context)
        case ast.Name(lineno=lineno, col_offset=col_offset, id=id):
            if id in context.current_functions:
                return stage_context_call("get_recursive_raw", {
                    "module": ast.Name(id="__name__", ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                    "func": ast.Constant(value=id, lineno=lineno, col_offset=col_offset)
                }, lineno, col_offset, context)
            else:
                return stage_context_call("get_by_name",
                                          {"val": expr, "name": ast.Constant(id, lineno=lineno, col_offset=col_offset)},
                                          lineno, col_offset, context)
        case ast.Call(func=func, args=args, keywords=keywords, lineno=lineno, col_offset=col_offset):

            # print("keywords", keywords)
            staged_args_ = []
            for expr in args:
                match expr:
                    case ast.Starred(value=value, lineno=lineno, col_offset=col_offset, ctx=ast.Load()):
                        unpack_call = stage_context_call("const_unpack", {"val": stage_expr(value, context)},
                                                         lineno, col_offset, context)
                        tmp_var = get_tmp_name()
                        context.block.append(
                            ast.Assign(
                                targets=[ast.Name(id=tmp_var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset)],
                                value=unpack_call, lineno=lineno, col_offset=col_offset))
                        staged_args_.append(ast.Starred(
                            value=ast.Name(id=tmp_var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                            lineno=lineno, col_offset=col_offset, ctx=ast.Load()))

                    case _:
                        staged_args_.append(stage_expr(expr, context))
            staged_args = ast.List(elts=staged_args_, ctx=ast.Load(), lineno=lineno, col_offset=col_offset)

            return stage_context_call("perform_call",
                                      {
                                          "fn": stage_expr(func, context),
                                          "args": staged_args,
                                          "kwargs": ast.Dict(
                                              keys=[ast.Constant(value=k.arg, lineno=lineno, col_offset=col_offset) for
                                                    k in
                                                    keywords], values=[stage_expr(k.value, context) for k in keywords],
                                              lineno=lineno, col_offset=col_offset)
                                      }, lineno,
                                      col_offset, context)
        case ast.BinOp(left=left, op=op, right=right, lineno=lineno, col_offset=col_offset):
            left_method, right_method = get_binop_methods(op)

            return stage_context_call("perform_binop",
                                      {
                                          "left": stage_expr(left, context),
                                          "right": stage_expr(right, context),
                                          "left_method": ast.Constant(value=left_method, lineno=lineno,
                                                                      col_offset=col_offset),
                                          "right_method": ast.Constant(value=right_method, lineno=lineno,
                                                                       col_offset=col_offset)
                                      }, lineno, col_offset, context)
        case ast.BoolOp(op=op, values=[l, r], lineno=lineno, col_offset=col_offset):
            match op:
                case ast.And():
                    return stage_context_call("bool_and", {
                        "left": stage_expr(l, context),
                        "right": stage_expr(r, context)
                    }, lineno, col_offset, context)
                case ast.Or():
                    return stage_context_call("bool_or", {
                        "left": stage_expr(l, context),
                        "right": stage_expr(r, context)
                    }, lineno, col_offset, context)
                case _:
                    raise NotImplementedError()
        case ast.BoolOp(op=op, values=values, lineno=lineno, col_offset=col_offset):
            val = values[0]
            for v in values[1:]:
                val = ast.BoolOp(op=op, values=[val, v], lineno=lineno, col_offset=col_offset)
            return stage_expr(val, context)

        case ast.Compare(left=left, ops=[ast.Is()], comparators=[right], lineno=lineno, col_offset=col_offset):
            return stage_context_call("is_", {
                "left": stage_expr(left, context),
                "right": stage_expr(right, context)
            }, lineno, col_offset, context)
        case ast.Compare(left=left, ops=[ast.IsNot()], comparators=[right], lineno=lineno, col_offset=col_offset):
            return stage_expr(ast.UnaryOp(op=ast.Not(),
                                          operand=ast.Compare(left=left, ops=[ast.Is()], comparators=[right],
                                                              lineno=lineno, col_offset=col_offset), lineno=lineno,
                                          col_offset=col_offset), context)
        case ast.Compare(left=left, ops=[ast.NotIn()], comparators=[right], lineno=lineno, col_offset=col_offset):
            return stage_expr(ast.UnaryOp(op=ast.Not(),
                                          operand=ast.Compare(left=left, ops=[ast.In()], comparators=[right],
                                                              lineno=lineno, col_offset=col_offset), lineno=lineno,
                                          col_offset=col_offset), context)
        case ast.Compare(left=left, ops=[ast.In()], comparators=[right], lineno=lineno, col_offset=col_offset):
            return stage_context_call("in_", {
                "left": stage_expr(left, context),
                "right": stage_expr(right, context)
            }, lineno, col_offset, context)
        case ast.Compare(left=left, ops=[op], comparators=[right], lineno=lineno, col_offset=col_offset):
            left_method, right_method = get_binop_methods(op)

            return stage_context_call("perform_binop",
                                      {
                                          "left": stage_expr(left, context),
                                          "right": stage_expr(right, context),
                                          "left_method": ast.Constant(value=left_method, lineno=lineno,
                                                                      col_offset=col_offset),
                                          "right_method": ast.Constant(value=right_method, lineno=lineno,
                                                                       col_offset=col_offset)
                                      }, lineno, col_offset, context)
        case ast.Compare(left=left, ops=op_list, comparators=comp_list, lineno=lineno, col_offset=col_offset):
            comparisons = []
            last = left
            for i in range(len(op_list)):
                comparisons.append(ast.Compare(left=last, ops=[op_list[i]], comparators=[comp_list[i]], lineno=lineno,
                                               col_offset=col_offset))
                last = comp_list[i]
            return stage_expr(ast.BoolOp(op=ast.And(), values=comparisons, lineno=lineno, col_offset=col_offset),
                              context)
        case ast.UnaryOp(op=op, operand=operand, lineno=lineno, col_offset=col_offset):
            match op:
                case ast.Not():
                    return stage_context_call("bool_not", {
                        "arg": stage_expr(operand, context)
                    }, lineno, col_offset, context)
                case ast.USub():
                    return stage_context_call("neg_", {
                        "arg": stage_expr(operand, context)
                    }, lineno, col_offset, context)
                case ast.Invert():
                    return stage_context_call("invert_", {
                        "arg": stage_expr(operand, context)
                    }, lineno, col_offset, context)
                case _:
                    raise NotImplementedError()
            # case ast.Constant(value=val, lineno=lineno, col_offset=col_offset):
            #    return ast.Call()
        case ast.IfExp(test=test, body=body, orelse=orelse, lineno=lineno, col_offset=col_offset):
            available_variables = context.available_variables
            body_analyzer = VariableAnalyzer()
            else_analyzer = VariableAnalyzer()
            body_analyzer.visit(body)
            else_analyzer.visit(orelse)
            required_variables = list(
                (body_analyzer.read_variables.union(else_analyzer.read_variables)).intersection(available_variables))
            tmp_variable = get_tmp_name()
            changed_variables = [tmp_variable]
            if_fn = stage_tmp_function([ast.Assign(
                targets=[ast.Name(id=tmp_variable, lineno=lineno, col_offset=col_offset, ctx=ast.Store())], value=body,
                lineno=lineno, col_offset=col_offset)], required_variables, changed_variables, context, lineno,
                col_offset)
            else_fn = stage_tmp_function([ast.Assign(
                targets=[ast.Name(id=tmp_variable, lineno=lineno, col_offset=col_offset, ctx=ast.Store())],
                value=orelse, lineno=lineno, col_offset=col_offset)], required_variables, changed_variables, context,
                lineno, col_offset)
            return ast.Subscript(value=stage_context_call("_if", {
                "cond": stage_expr(test, context), "bodyfn": if_fn, "elsefn": else_fn, "inputs": ast.List(
                    elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                          required_variables],
                    ctx=ast.Load(), lineno=lineno, col_offset=col_offset)
            }, lineno, col_offset, context), slice=ast.Constant(value=0, lineno=lineno, col_offset=col_offset),
                                 lineno=lineno,
                                 col_offset=col_offset, ctx=ast.Load())
        case ast.Lambda(args=args, body=body, lineno=lineno, col_offset=col_offset):
            available_variables = context.available_variables
            lambda_arg_names = [arg.arg for arg in args.args]  # todo: handle other types of args

            binder = BindFreeVars(available_variables, lambda_arg_names, lineno, col_offset)
            implicit_closure_body = copy.deepcopy(body)
            explicit_closure_body = binder.visit(body)

            global lambda_fn_counter
            lambda_fn_counter += 1
            lambda_funcname = f"__hipy_lambda_fn{lambda_fn_counter}"
            lambda_funcname_explicit = f"__hipy_lambda_fn_explicit{lambda_fn_counter}"
            lambda_funcname_explicit_staged = f"__hipy_lambda_fn_explicit_staged{lambda_fn_counter}"
            explicit_args = copy.deepcopy(args)
            explicit_args.kwonlyargs.append(
                ast.arg(arg="__closure__", annotation=None, lineno=lineno, col_offset=col_offset))
            explicit_args.kw_defaults.append(ast.Constant(value=None, lineno=lineno, col_offset=col_offset))

            explicit_funcdef = ast.FunctionDef(name=lambda_funcname_explicit, args=explicit_args,
                                               body=[ast.Return(value=explicit_closure_body, lineno=lineno,
                                                                col_offset=col_offset)],
                                               lineno=lineno, col_offset=col_offset, decorator_list=[])
            explicit_staged_funcdef = ast.FunctionDef(name=lambda_funcname_explicit_staged, args=explicit_args,
                                                      body=[ast.Return(value=explicit_closure_body, lineno=lineno,
                                                                       col_offset=col_offset)],
                                                      lineno=lineno, col_offset=col_offset, decorator_list=[])

            staged_funcdef = ast.FunctionDef(name=lambda_funcname, args=args,
                                             body=[ast.Return(value=implicit_closure_body, lineno=lineno,
                                                              col_offset=col_offset)],
                                             lineno=lineno, col_offset=col_offset, decorator_list=[])
            staged_funcdef = stage_function(staged_funcdef, context.globals, nested=False, outer_context=context)
            explicit_staged_funcdef = stage_function(explicit_staged_funcdef, context.globals, nested=False,
                                                     outer_context=context)
            context.block.append(explicit_staged_funcdef)
            context.block.append(staged_funcdef)
            closure_var_list = list(binder.ids)
            closure_obj = ast.Dict(
                keys=[ast.Constant(value=var, lineno=lineno, col_offset=col_offset) for var in closure_var_list],
                values=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                        closure_var_list], lineno=lineno, col_offset=col_offset)

            explicit_staged_funcref = ast.Name(id=lambda_funcname_explicit_staged, ctx=ast.Load(), lineno=lineno,
                                               col_offset=col_offset)
            bind_python_lambda = ast.Lambda(
                args=ast.arguments(
                    args=[ast.arg(arg='__hipy_fn__', annotation=None, lineno=lineno, col_offset=col_offset)],
                    posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=ast.Call(func=ast.Name(id='__hipy_fn__', ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                              args=[closure_obj,
                                    ast.Constant(value=ast.unparse(explicit_funcdef), lineno=lineno,
                                                 col_offset=col_offset),
                                    ast.Constant(value=lambda_funcname_explicit, lineno=lineno, col_offset=col_offset)],
                              keywords=[],
                              lineno=lineno, col_offset=col_offset), lineno=lineno,
                col_offset=col_offset)
            bind_staged_lambda = ast.Lambda(
                args=ast.arguments(
                    args=[ast.arg(arg='__hipy_fn__', annotation=None, lineno=lineno, col_offset=col_offset)],
                    posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=ast.Call(func=ast.Name(id='__hipy_fn__', ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                              args=[closure_obj, explicit_staged_funcref], keywords=[], lineno=lineno,
                              col_offset=col_offset),
                lineno=lineno,
                col_offset=col_offset)
            return stage_context_call("create_lambda", {
                "staged": ast.Name(id=lambda_funcname, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                "bind_python": bind_python_lambda,
                "bind_staged": bind_staged_lambda}, lineno, col_offset, context)
        case _:
            print("unhandled expr", type(expr))
            raise NotImplementedError()


class VariableAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.read_variables = set()
        self.written_variables = set()
        self.not_track_write=0

    def visit_Name(self, node):
        self.read_variables.add(node.id)
        if isinstance(node.ctx, ast.Store):
            if self.not_track_write==0:
                self.written_variables.add(node.id)


    def handle_import_alias(self, alias):
        match alias:
            case ast.alias(name=name, asname=asname):
                if asname is None:
                    self.written_variables.add(name)
                else:
                    self.written_variables.add(asname)

    def visit_Import(self, node):
        for alias in node.names:
            self.handle_import_alias(alias)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.handle_import_alias(alias)
    def visit_ListComp(self, node : ast.ListComp):
        self.not_track_write+=1
        self.visit(node.elt)
        for g in node.generators:
            self.visit(g)

        self.not_track_write-=1


tmp_function_counter = 0


def plain_function(body, input_vars, context: StageContext, lineno, col_offset):
    global tmp_function_counter
    tmp_function_counter += 1
    name = f"tmp_function_{tmp_function_counter}"
    args = ast.arguments(
        args=[ast.arg(arg=arg, annotation=None, lineno=lineno, col_offset=col_offset) for arg in input_vars],
        posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    fn = stage_function(ast.FunctionDef(name=name, args=args, body=body, lineno=lineno, col_offset=col_offset,
                                        decorator_list=[]), context.globals, nested=False, outer_context=context)
    context.block.append(fn)
    return ast.Name(id=name, ctx=ast.Load(), lineno=lineno, col_offset=col_offset)


def stage_tmp_function(body, input_vars, output_vars, context: StageContext, lineno, col_offset):
    global tmp_function_counter
    tmp_function_counter += 1
    name = f"tmp_function_{tmp_function_counter}"
    args = ast.arguments(
        args=[ast.arg(arg=arg, annotation=None, lineno=lineno, col_offset=col_offset) for arg in input_vars],
        posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    fn = stage_function(ast.FunctionDef(name=name, args=args, body=body, lineno=lineno, col_offset=col_offset,
                                        decorator_list=[]), context.globals, nested=True, outer_context=context)
    fn.body.append(ast.Return(
        value=ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                              output_vars], ctx=ast.Load(), lineno=lineno, col_offset=col_offset), lineno=lineno,
        col_offset=col_offset))
    context.block.append(fn)
    return ast.Name(id=name, ctx=ast.Load(), lineno=lineno, col_offset=col_offset)


class Target(abc.ABC):
    @abc.abstractmethod
    def assign(self, value, context: StageContext):
        pass

    @abc.abstractmethod
    def augassign(self, fn, context: StageContext):
        pass


class AttributeTarget(Target):
    def __init__(self, target, attr, lineno, col_offset):
        self.target = target
        self.attr = attr
        self.lineno = lineno
        self.col_offset = col_offset

    def assign(self, value, context: StageContext):
        context.block.append(ast.Expr(value=stage_context_call("set_attr", {
            "val": stage_expr(self.target, context),
            "attr": ast.Constant(value=self.attr, lineno=self.lineno, col_offset=self.col_offset),
            "value": value
        }, self.lineno, self.col_offset, context), lineno=self.lineno, col_offset=self.col_offset))

    def augassign(self, fn, context: StageContext):
        target_tmp_var = get_tmp_name()
        context.block.append(
            ast.Assign(
                targets=[ast.Name(id=target_tmp_var, ctx=ast.Store(), lineno=self.lineno, col_offset=self.col_offset)],
                value=stage_expr(self.target, context), lineno=self.lineno, col_offset=self.col_offset))
        loaded = stage_context_call("get_attr", {
            "val": ast.Name(id=target_tmp_var, ctx=ast.Load(), lineno=self.lineno, col_offset=self.col_offset),
            "attr": ast.Constant(value=self.attr, lineno=self.lineno, col_offset=self.col_offset),
        }, self.lineno, self.col_offset, context)
        context.block.append(ast.Expr(value=stage_context_call("set_attr", {
            "val": ast.Name(id=target_tmp_var, ctx=ast.Load(), lineno=self.lineno, col_offset=self.col_offset),
            "attr": ast.Constant(value=self.attr, lineno=self.lineno, col_offset=self.col_offset),
            "value": fn(loaded)
        }, self.lineno, self.col_offset, context), lineno=self.lineno, col_offset=self.col_offset))


class NameTarget(Target):
    def __init__(self, name, lineno, col_offset):
        self.name = name
        self.lineno = lineno
        self.col_offset = col_offset

    def assign(self, value, context: StageContext):
        context.block.append(ast.Assign(targets=[ast.Name(id=self.name, lineno=self.lineno,
                                                          col_offset=self.col_offset, ctx=ast.Store())], value=value,
                                        lineno=self.lineno,
                                        col_offset=self.col_offset))
        context.available_variables.add(self.name)

    def augassign(self, fn, context: StageContext):
        context.block.append(ast.Assign(
            targets=[ast.Name(id=self.name, lineno=self.lineno, col_offset=self.col_offset, ctx=ast.Store())],
            value=fn(ast.Name(id=self.name, lineno=self.lineno, col_offset=self.col_offset, ctx=ast.Load())),
            lineno=self.lineno, col_offset=self.col_offset))


class SubscriptTarget(Target):
    def __init__(self, target, slice, lineno, col_offset):
        self.target = target
        self.slice = slice
        self.lineno = lineno
        self.col_offset = col_offset

    def assign(self, value, context: StageContext):
        context.block.append(ast.Expr(value=stage_context_call("set_item", {
            "val": stage_expr(self.target, context),
            "item": stage_expr(self.slice, context),
            "value": value
        }, self.lineno, self.col_offset, context), lineno=self.lineno, col_offset=self.col_offset))

    def augassign(self, fn, context: StageContext):
        target_tmp_var = get_tmp_name()
        context.block.append(
            ast.Assign(
                targets=[ast.Name(id=target_tmp_var, ctx=ast.Store(), lineno=self.lineno, col_offset=self.col_offset)],
                value=stage_expr(self.target, context), lineno=self.lineno, col_offset=self.col_offset))
        slice_tmp_var = get_tmp_name()
        context.block.append(
            ast.Assign(
                targets=[ast.Name(id=slice_tmp_var, ctx=ast.Store(), lineno=self.lineno, col_offset=self.col_offset)],
                value=stage_expr(self.slice, context), lineno=self.lineno, col_offset=self.col_offset))
        loaded = stage_context_call("get_item", {
            "val": ast.Name(id=target_tmp_var, ctx=ast.Load(), lineno=self.lineno, col_offset=self.col_offset),
            "item": ast.Name(id=slice_tmp_var, ctx=ast.Load(), lineno=self.lineno, col_offset=self.col_offset)
        }, self.lineno, self.col_offset, context)
        context.block.append(ast.Expr(value=stage_context_call("set_item", {
            "val": ast.Name(id=target_tmp_var, ctx=ast.Load(), lineno=self.lineno, col_offset=self.col_offset),
            "item": ast.Name(id=slice_tmp_var, ctx=ast.Load(), lineno=self.lineno, col_offset=self.col_offset),
            "value": fn(loaded)
        }, self.lineno, self.col_offset, context), lineno=self.lineno, col_offset=self.col_offset))


class TupleTarget(Target):
    def __init__(self, children, lineno, col_offset):
        self.children = children
        self.lineno = lineno
        self.col_offset = col_offset

    def assign(self, value, context: StageContext):
        lineno = self.lineno
        col_offset = self.col_offset
        unpack_call = stage_context_call("unpack", {"val": value,
                                                    "num": ast.Constant(value=len(self.children), lineno=lineno,
                                                                        col_offset=col_offset)},
                                         lineno, col_offset, context)
        tmp_var = get_tmp_name()
        context.block.append(
            ast.Assign(targets=[ast.Name(id=tmp_var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset)],
                       value=unpack_call, lineno=lineno, col_offset=col_offset))
        for i, target in enumerate(self.children):
            unpacked_value = ast.Subscript(
                value=ast.Name(id=tmp_var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                slice=ast.Constant(value=i, lineno=lineno, col_offset=col_offset), ctx=ast.Load(),
                lineno=lineno, col_offset=col_offset)
            target.assign(unpacked_value, context)

    def augassign(self, fn, context: StageContext):
        raise NotImplementedError()


def createTarget(target):
    match target:
        case ast.Name(id=name, lineno=lineno, col_offset=col_offset):
            return NameTarget(name, lineno, col_offset)
        case ast.Attribute(value=val, attr=attr, lineno=lineno, col_offset=col_offset):
            return AttributeTarget(val, attr, lineno, col_offset)
        case ast.Subscript(value=val, slice=slice, lineno=lineno, col_offset=col_offset):
            return SubscriptTarget(val, slice, lineno, col_offset)
        case ast.Tuple(elts=elts, lineno=lineno, col_offset=col_offset):
            return TupleTarget([createTarget(elt) for elt in elts], lineno, col_offset)
        case _:
            raise NotImplementedError()


def stage_stmt(stmt, context: StageContext):
    global lambda_fn_counter
    match stmt:
        case ast.Expr(value=expr, lineno=lineno, col_offset=col_offset):
            context.block.append(ast.Expr(value=stage_expr(expr, context), lineno=lineno, col_offset=col_offset))
        case ast.Assign(targets=[target], value=value, lineno=lineno, col_offset=col_offset):
            createTarget(target).assign(stage_expr(value, context), context)
        case ast.AnnAssign(target=target, value=value, lineno=lineno, col_offset=col_offset):
            createTarget(target).assign(stage_expr(value, context), context)
        case ast.AugAssign(target=target, op=op, value=value, lineno=lineno, col_offset=col_offset):

            createTarget(target).augassign(lambda x: stage_context_call("perform_call", {
                "fn": stage_context_call("get_attr", {
                    "val": x,
                    "attr": ast.Constant(value=get_aug_method(op), lineno=lineno, col_offset=col_offset),
                }, lineno, col_offset, context),
                "args": ast.List(elts=[stage_expr(value, context)], ctx=ast.Load(), lineno=lineno,
                                 col_offset=col_offset),
            }, lineno, col_offset, context), context)

        case ast.If(test=test, body=body, orelse=orelse, lineno=lineno, col_offset=col_offset):
            available_variables = context.available_variables
            body_analyzer = VariableAnalyzer()
            else_analyzer = VariableAnalyzer()
            for body_stmt in body:
                body_analyzer.visit(body_stmt)
            for else_stmt in orelse:
                else_analyzer.visit(else_stmt)
            changed_if = body_analyzer.written_variables.intersection(available_variables)
            new_if = body_analyzer.written_variables - available_variables
            changed_else = else_analyzer.written_variables.intersection(available_variables)
            new_else = else_analyzer.written_variables - available_variables
            changed_variables = list(changed_if.union(changed_else).union(new_if.intersection(new_else)))
            required_variables = list(
                (body_analyzer.read_variables.union(else_analyzer.read_variables)).intersection(available_variables))
            required_variables += [x for x in changed_variables if
                                   x not in required_variables and x in available_variables]

            if_fn = stage_tmp_function(body, required_variables, changed_variables, context, lineno, col_offset)
            else_fn = stage_tmp_function(orelse, required_variables, changed_variables, context, lineno, col_offset)
            raw_res = stage_context_call("_if", {
                "cond": stage_expr(test, context), "bodyfn": if_fn, "elsefn": else_fn, "inputs": ast.List(
                    elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                          required_variables],
                    ctx=ast.Load(), lineno=lineno, col_offset=col_offset)
            }, lineno, col_offset, context)
            assignment = ast.Assign(
                targets=[ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset) for var
                                         in changed_variables], ctx=ast.Store(), lineno=lineno,
                                   col_offset=col_offset)], value=raw_res, lineno=lineno,
                col_offset=col_offset)
            early_return_val = ast.Attribute(
                value=ast.Name(id='e', ctx=ast.Load(), lineno=lineno, col_offset=col_offset), attr='val',
                ctx=ast.Load(), lineno=lineno, col_offset=col_offset)
            if context.nested:
                early_return = ast.Expr(value=stage_context_call("early_return", {"val": early_return_val}, lineno,
                                                                 col_offset, context), lineno=lineno,
                                        col_offset=col_offset)
            else:
                early_return = ast.Return(value=early_return_val, lineno=lineno, col_offset=col_offset)
            context.block.append(
                ast.Try(body=[assignment], handlers=[
                    ast.ExceptHandler(
                        type=ast.Attribute(
                            ast.Name(id='_context', ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                            attr='EarlyReturn', ctx=ast.Load(), lineno=lineno, col_offset=col_offset), name='e',
                        body=[early_return], lineno=lineno, col_offset=col_offset)], orelse=[], finalbody=[],
                        lineno=lineno, col_offset=col_offset)
            )
        case ast.Return(value=value, lineno=lineno, col_offset=col_offset):
            if context.nested:
                if value is None:
                    value = ast.Constant(value=None, lineno=lineno, col_offset=col_offset)
                context.block.append(
                    ast.Expr(value=stage_context_call("early_return", {"val": stage_expr(value, context)}, lineno,
                                                      col_offset, context), lineno=lineno, col_offset=col_offset))
            else:
                context.block.append(ast.Return(value=stage_expr(value, context), lineno=lineno, col_offset=col_offset))
        case ast.While(test=test, body=body, orelse=[], lineno=lineno, col_offset=col_offset):

            available_variables = context.available_variables
            body_analyzer = VariableAnalyzer()
            for body_stmt in body:
                body_analyzer.visit(body_stmt)
            body_analyzer.visit(test)
            changed_variables = body_analyzer.written_variables.intersection(available_variables)
            read_variables = body_analyzer.read_variables.intersection(available_variables)
            read_only_inputs = list(read_variables - changed_variables)
            iter_vals = list(changed_variables)
            required_variables = read_only_inputs + iter_vals
            body_fn = stage_tmp_function(body, required_variables, changed_variables, context, lineno, col_offset)
            test_fn = plain_function([ast.Return(test, lineno=lineno, col_offset=col_offset)], required_variables,
                                     context, lineno, col_offset)
            raw_res = stage_context_call("_while", {
                "cond": test_fn,
                "body": body_fn,
                "read_only_inputs": ast.List(
                    elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                          read_only_inputs],
                    ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                "iter_vals": ast.List(
                    elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in iter_vals],
                    ctx=ast.Load(), lineno=lineno, col_offset=col_offset)
            }, lineno, col_offset, context)
            assignment = ast.Assign(
                targets=[ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset) for var
                                         in changed_variables], ctx=ast.Store(), lineno=lineno,
                                   col_offset=col_offset)], value=raw_res, lineno=lineno,
                col_offset=col_offset)
            context.block.append(assignment)
        case ast.For(target=target, iter=over, body=body, orelse=[], lineno=lineno,
                     col_offset=col_offset):

            available_variables = context.available_variables
            body_analyzer = VariableAnalyzer()
            for body_stmt in body:
                body_analyzer.visit(body_stmt)
            changed_variables = body_analyzer.written_variables.intersection(available_variables)
            read_variables = body_analyzer.read_variables.intersection(available_variables)
            read_only_inputs = list(read_variables - changed_variables)
            iter_vals = list(changed_variables)
            required_variables = read_only_inputs + iter_vals
            targetname = get_tmp_name()
            body = [ast.Assign(targets=[target],
                               value=ast.Name(id=targetname, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                               lineno=lineno, col_offset=col_offset)] + body
            body_fn = stage_tmp_function(body, required_variables + [targetname], changed_variables, context, lineno,
                                         col_offset)
            raw_res = stage_context_call("_for", {
                "over": stage_expr(over, context),
                "target": ast.Constant(value=targetname, lineno=lineno, col_offset=col_offset),
                "body": body_fn,
                "read_only_inputs": ast.List(
                    elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                          read_only_inputs],
                    ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                "iter_vals": ast.List(
                    elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in iter_vals],
                    ctx=ast.Load(), lineno=lineno, col_offset=col_offset)
            }, lineno, col_offset, context)
            assignment = ast.Assign(
                targets=[ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset) for var
                                         in changed_variables], ctx=ast.Store(), lineno=lineno,
                                   col_offset=col_offset)], value=raw_res, lineno=lineno,
                col_offset=col_offset)
            context.block.append(assignment)

        case ast.Try(body=body, handlers=[ast.ExceptHandler(type=None, name=None, body=except_body)]):
            lineno = 0
            col_offset = 0
            available_variables = context.available_variables
            body_analyzer = VariableAnalyzer()
            else_analyzer = VariableAnalyzer()
            for body_stmt in body:
                body_analyzer.visit(body_stmt)
            for except_stmt in except_body:
                else_analyzer.visit(except_stmt)
            changed_try = body_analyzer.written_variables.intersection(available_variables)
            new_try = body_analyzer.written_variables - available_variables
            changed_except = else_analyzer.written_variables.intersection(available_variables)
            new_except = else_analyzer.written_variables - available_variables
            changed_variables = list(changed_try.union(changed_except).union(new_try.intersection(new_except)))
            try_binder = BindFreeVars(available_variables, new_try, lineno, col_offset)
            except_binder = BindFreeVars(available_variables, new_except, lineno, col_offset)

            explicit_try_body = list(map(lambda n: try_binder.visit(n), body))
            explicit_except_body = list(map(lambda n: except_binder.visit(n), except_body))
            lambda_fn_counter += 1
            lambda_funcname_try = f"__hipy_lambda_try{lambda_fn_counter}"
            lambda_funcname_except = f"__hipy_lambda_except{lambda_fn_counter}"
            # explicit_args.kwonlyargs.append(ast.arg(arg="__closure__", annotation=None, lineno=lineno, col_offset=col_offset))
            # explicit_args.kw_defaults.append(ast.Constant(value=None, lineno=lineno, col_offset=col_offset))
            try_funcdef = ast.FunctionDef(name=lambda_funcname_try,
                                          args=ast.arguments(args=[], posonlyargs=[], vararg=None, kwonlyargs=[
                                              ast.arg(arg="__closure__", annotation=None, lineno=lineno,
                                                      col_offset=col_offset)], kw_defaults=[
                                              ast.Constant(value=None, lineno=lineno, col_offset=col_offset)],
                                                             kwarg=None, defaults=[]), body=explicit_try_body,
                                          lineno=lineno,
                                          col_offset=col_offset, decorator_list=[])
            try_funcdef_staged = stage_function(try_funcdef, context.globals, nested=True, outer_context=context)
            try_funcdef_staged.body.append(ast.Return(
                value=ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                                      changed_variables], ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                lineno=lineno,
                col_offset=col_offset))
            context.block.append(try_funcdef_staged)
            except_funcdef = ast.FunctionDef(name=lambda_funcname_except,
                                             args=ast.arguments(args=[], posonlyargs=[], vararg=None, kwonlyargs=[
                                                 ast.arg(arg="__closure__", annotation=None, lineno=lineno,
                                                         col_offset=col_offset)], kw_defaults=[
                                                 ast.Constant(value=None, lineno=lineno, col_offset=col_offset)],
                                                                kwarg=None, defaults=[]), body=explicit_except_body,
                                             lineno=lineno,
                                             col_offset=col_offset, decorator_list=[])
            except_funcdef_staged = stage_function(except_funcdef, context.globals, nested=True, outer_context=context)
            except_funcdef_staged.body.append(ast.Return(
                value=ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                                      changed_variables], ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                lineno=lineno,
                col_offset=col_offset))
            context.block.append(except_funcdef_staged)
            try_closure_var_list = list(try_binder.ids)
            try_closure_obj = ast.Dict(
                keys=[ast.Constant(value=var, lineno=lineno, col_offset=col_offset) for var in try_closure_var_list],
                values=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                        try_closure_var_list], lineno=lineno, col_offset=col_offset)
            except_closure_var_list = list(except_binder.ids)
            except_closure_obj = ast.Dict(
                keys=[ast.Constant(value=var, lineno=lineno, col_offset=col_offset) for var in except_closure_var_list],
                values=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                        except_closure_var_list], lineno=lineno, col_offset=col_offset)
            raw_res = stage_context_call("_try", {
                "tryfn": ast.Name(id=lambda_funcname_try, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                "exceptfn": ast.Name(id=lambda_funcname_except, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                "try_closure": try_closure_obj,
                "except_closure": except_closure_obj,
            }, lineno, col_offset, context)
            assignment = ast.Assign(
                targets=[ast.Tuple(elts=[ast.Name(id=var, ctx=ast.Store(), lineno=lineno, col_offset=col_offset) for var
                                         in changed_variables], ctx=ast.Store(), lineno=lineno,
                                   col_offset=col_offset)], value=raw_res, lineno=lineno,
                col_offset=col_offset)
            early_return_val = ast.Attribute(
                value=ast.Name(id='e', ctx=ast.Load(), lineno=lineno, col_offset=col_offset), attr='val',
                ctx=ast.Load(), lineno=lineno, col_offset=col_offset)
            if context.nested:
                early_return = ast.Expr(value=stage_context_call("early_return", {"val": early_return_val}, lineno,
                                                                 col_offset, context), lineno=lineno,
                                        col_offset=col_offset)
            else:
                early_return = ast.Return(value=early_return_val, lineno=lineno, col_offset=col_offset)
            context.block.append(
                ast.Try(body=[assignment], handlers=[
                    ast.ExceptHandler(
                        type=ast.Attribute(
                            ast.Name(id='_context', ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                            attr='EarlyReturn', ctx=ast.Load(), lineno=lineno, col_offset=col_offset), name='e',
                        body=[early_return], lineno=lineno, col_offset=col_offset)], orelse=[], finalbody=[],
                        lineno=lineno, col_offset=col_offset)
            )

        case ast.Pass():
            context.block.append(stmt)

        case ast.FunctionDef(name=name, args=args, body=body, decorator_list=decorators, returns=returns, lineno=lineno,
                             col_offset=col_offset):
            available_variables = context.available_variables
            lambda_arg_names = [arg.arg for arg in args.args]  # todo: handle other types of args

            binder = BindFreeVars(available_variables, lambda_arg_names, lineno, col_offset)
            implicit_closure_body = copy.deepcopy(body)
            explicit_closure_body = list(map(lambda n: binder.visit(n), body))

            lambda_fn_counter += 1
            lambda_funcname = f"__hipy_lambda_fn{lambda_fn_counter}"
            lambda_funcname_explicit = f"__hipy_lambda_fn_explicit{lambda_fn_counter}"
            lambda_funcname_explicit_staged = f"__hipy_lambda_fn_explicit_staged{lambda_fn_counter}"
            explicit_args = copy.deepcopy(args)
            explicit_args.kwonlyargs.append(
                ast.arg(arg="__closure__", annotation=None, lineno=lineno, col_offset=col_offset))
            explicit_args.kw_defaults.append(ast.Constant(value=None, lineno=lineno, col_offset=col_offset))

            explicit_funcdef = ast.FunctionDef(name=lambda_funcname_explicit, args=explicit_args,
                                               body=explicit_closure_body,
                                               lineno=lineno, col_offset=col_offset, decorator_list=[])
            explicit_staged_funcdef = ast.FunctionDef(name=lambda_funcname_explicit_staged, args=explicit_args,
                                                      body=explicit_closure_body,
                                                      lineno=lineno, col_offset=col_offset, decorator_list=[])

            staged_funcdef = ast.FunctionDef(name=lambda_funcname, args=args,
                                             body=implicit_closure_body,
                                             lineno=lineno, col_offset=col_offset, decorator_list=[])
            staged_funcdef = stage_function(staged_funcdef, context.globals, nested=False, outer_context=context)
            explicit_staged_funcdef = stage_function(explicit_staged_funcdef, context.globals, nested=False,
                                                     outer_context=context)
            context.block.append(explicit_staged_funcdef)
            context.block.append(staged_funcdef)
            closure_var_list = list(binder.ids)
            closure_obj = ast.Dict(
                keys=[ast.Constant(value=var, lineno=lineno, col_offset=col_offset) for var in closure_var_list],
                values=[ast.Name(id=var, ctx=ast.Load(), lineno=lineno, col_offset=col_offset) for var in
                        closure_var_list], lineno=lineno, col_offset=col_offset)

            explicit_staged_funcref = ast.Name(id=lambda_funcname_explicit_staged, ctx=ast.Load(), lineno=lineno,
                                               col_offset=col_offset)
            bind_python_lambda = ast.Lambda(
                args=ast.arguments(
                    args=[ast.arg(arg='__hipy_fn__', annotation=None, lineno=lineno, col_offset=col_offset)],
                    posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=ast.Call(func=ast.Name(id='__hipy_fn__', ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                              args=[closure_obj,
                                    ast.Constant(value=ast.unparse(explicit_funcdef), lineno=lineno,
                                                 col_offset=col_offset),
                                    ast.Constant(value=lambda_funcname_explicit, lineno=lineno, col_offset=col_offset)],
                              keywords=[],
                              lineno=lineno, col_offset=col_offset), lineno=lineno,
                col_offset=col_offset)
            bind_staged_lambda = ast.Lambda(
                args=ast.arguments(
                    args=[ast.arg(arg='__hipy_fn__', annotation=None, lineno=lineno, col_offset=col_offset)],
                    posonlyargs=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
                body=ast.Call(func=ast.Name(id='__hipy_fn__', ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                              args=[closure_obj, explicit_staged_funcref], keywords=[], lineno=lineno,
                              col_offset=col_offset),
                lineno=lineno,
                col_offset=col_offset)
            lambda_val = stage_context_call("create_lambda", {
                "staged": ast.Name(id=lambda_funcname, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                "bind_python": bind_python_lambda,
                "bind_staged": bind_staged_lambda}, lineno, col_offset, context)
            context.block.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store(), lineno=lineno,
                                                              col_offset=col_offset)], value=lambda_val, lineno=lineno,
                                            col_offset=col_offset))
            context.available_variables.add(name)
        case _:
            print("unhandled stmt", type(stmt))
            raise NotImplementedError()


def stage_block(block, context: StageContext):
    for stmt in block:
        stage_stmt(stmt, context)


def rewrite_if_return(body):
    res_body = []
    for i, stmt in enumerate(body):
        match stmt:
            case ast.If(test=test, body=ifBody, orelse=[], lineno=lineno, col_offset=col_offset):
                if isinstance(ifBody[-1], ast.Return):
                    remainder = body[i + 1:]
                    if len(remainder) == 0 or not isinstance(remainder[-1], ast.Return):
                        remainder.append(ast.Return(value=None, lineno=lineno, col_offset=col_offset))
                    res_body.append(
                        ast.If(test=test, body=ifBody, orelse=remainder, lineno=lineno, col_offset=col_offset))
                    break
                else:
                    res_body.append(stmt)
            case _:
                res_body.append(stmt)
    return res_body


def flatten(l):
    r = []
    for i in l:
        if isinstance(i, list):
            r += flatten(i)
        else:
            r.append(i)

    return r


def rewrite_continue_in_loop_body(body):
    res_body = []
    for stmt in reversed(body):
        match stmt:
            case ast.If(test=test, body=ifBody, orelse=[], lineno=lineno, col_offset=col_offset):
                if isinstance(ifBody[-1], ast.Continue):
                    res_body = [
                        ast.If(test=test, body=ifBody[:-1], orelse=res_body, lineno=lineno, col_offset=col_offset)]
                else:
                    res_body = [stmt] + res_body
            case _:
                res_body = [stmt] + res_body
    return res_body


def rewrite_loop_break(body):
    break_vars = 0

    class RewriteBreakInLoop(ast.NodeTransformer):
        def __init__(self):
            nonlocal break_vars
            super().__init__()
            self.variable_name = f"break_var_{break_vars}"
            break_vars += 1
            self.rewrote = False

        def visit_For(self, node):
            return node

        def visit_While(self, node):
            return node

        def visit_Break(self, node):
            self.rewrote = True
            return [
                ast.Assign(targets=[
                    ast.Name(id=self.variable_name, ctx=ast.Store(), lineno=node.lineno, col_offset=node.col_offset)],
                    value=ast.Constant(value=False, lineno=node.lineno, col_offset=node.col_offset),
                    lineno=node.lineno, col_offset=node.col_offset),
                ast.Continue(lineno=node.lineno, col_offset=node.col_offset)
            ]

    class RewriteLoopWithBreak(ast.NodeTransformer):
        def visit_For(self, node):
            rewriter = RewriteBreakInLoop()
            new_body = []
            for stmt in node.body:
                rewritten = rewriter.visit(stmt)
                match rewritten:
                    case list():
                        new_body += rewritten
                    case None:
                        pass
                    case _:
                        new_body.append(rewritten)

            if rewriter.rewrote:
                new_body = rewrite_continue_in_loop_body(new_body)
                encapsulated = ast.If(test=ast.Name(id=rewriter.variable_name, ctx=ast.Load(), lineno=node.lineno,
                                                    col_offset=node.col_offset), body=new_body, orelse=[],
                                      lineno=node.lineno, col_offset=node.col_offset)
                return [
                    ast.Assign(targets=[ast.Name(id=rewriter.variable_name, ctx=ast.Store(), lineno=node.lineno,
                                                 col_offset=node.col_offset)],
                               value=ast.Constant(value=True, lineno=node.lineno, col_offset=node.col_offset),
                               lineno=node.lineno, col_offset=node.col_offset),
                    ast.For(target=node.target, iter=node.iter, body=[encapsulated], orelse=node.orelse,
                            lineno=node.lineno, col_offset=node.col_offset)
                ]
            else:
                return node

    rewriter = RewriteLoopWithBreak()
    return flatten([rewriter.visit(stmt) for stmt in body])


def rewrite_loop_if_continue(body):
    class RewriteLoopIfContinue(ast.NodeTransformer):

        def visit_For(self, node):
            node.body = rewrite_continue_in_loop_body(node.body)
            return node

    rewriter = RewriteLoopIfContinue()
    return [rewriter.visit(stmt) for stmt in body]


def rewrite_func(body):
    body = rewrite_if_return(body)
    body = rewrite_loop_break(body)
    body = rewrite_loop_if_continue(body)
    return body


def stage_function(func, globals, nested=False, outer_context=None):
    match func:
        case ast.FunctionDef(name=name, args=args, body=body, decorator_list=decorators, returns=returns, lineno=lineno,
                             col_offset=col_offset):
            body = rewrite_func(body)
            res_body = []
            current_functions = (outer_context.current_functions | {name}) if outer_context is not None else {name}
            context = StageContext(res_body, globals, nested, current_functions)
            if outer_context and nested:
                context.available_variables = copy.deepcopy(outer_context.available_variables)
            for arg in args.args:
                context.available_variables.add(arg.arg)
            # todo: also other names might be available
            num_args = len(args.args)
            num_defaults = len(args.defaults)
            for i in range(num_args - num_defaults, num_args):
                default_expr = args.defaults[i - num_args + num_defaults]
                arg_name = args.args[i].arg
                args.defaults[i - num_args + num_defaults] = ast.Constant(value=None, lineno=lineno,
                                                                          col_offset=col_offset)
                need_default = ast.Compare(
                    left=ast.Name(id=arg_name, ctx=ast.Load(), lineno=lineno, col_offset=col_offset),
                    ops=[ast.Is()],
                    comparators=[ast.Constant(value=None, lineno=lineno, col_offset=col_offset)], lineno=lineno,
                    col_offset=col_offset)
                actual_value = ast.IfExp(test=need_default, body=stage_expr(default_expr, context),
                                         orelse=ast.Name(id=arg_name, ctx=ast.Load(), lineno=lineno,
                                                         col_offset=col_offset), lineno=lineno,
                                         col_offset=col_offset)
                res_body.append(
                    ast.Assign(targets=[ast.Name(id=arg_name, ctx=ast.Store(), lineno=lineno, col_offset=col_offset)],
                               value=actual_value, lineno=lineno, col_offset=col_offset))
            if args.vararg is not None:
                res_body.append(
                    ast.Assign(
                        targets=[ast.Name(id=args.vararg.arg, ctx=ast.Store(), lineno=lineno, col_offset=col_offset)],
                        value=stage_context_call("create_tuple", {
                            "elts": ast.Name(id=args.vararg.arg, ctx=ast.Load(), lineno=lineno, col_offset=col_offset)},
                                                 lineno, col_offset, context), lineno=lineno, col_offset=col_offset))
            if args.kwarg is not None:
                res_body.append(
                    ast.Assign(
                        targets=[ast.Name(id=args.kwarg.arg, ctx=ast.Store(), lineno=lineno, col_offset=col_offset)],
                        value=stage_context_call("create_dict_simple", {
                            "d": ast.Name(id=args.kwarg.arg, ctx=ast.Load(), lineno=lineno, col_offset=col_offset)},
                                                 lineno, col_offset, context), lineno=lineno, col_offset=col_offset))

            stage_block(body, context)
            args.kwonlyargs.append(ast.arg(arg="_context", annotation=None, lineno=lineno, col_offset=col_offset))
            args.kw_defaults.append(ast.Constant(value=None, lineno=lineno, col_offset=col_offset))

            return ast.FunctionDef(name=name, args=args, body=res_body, lineno=lineno, col_offset=col_offset,
                                   decorator_list=[])
        case _:
            raise NotImplementedError()


class LineNumberAdapter(ast.NodeVisitor):
    def __init__(self, line_offset, col_offset):
        self.line_offset = line_offset
        self.col_offset = col_offset

    def visit(self, node):
        # If the node has a lineno attribute, adjust it by the line offset
        if hasattr(node, 'lineno'):
            node.lineno += self.line_offset - 1
            node.end_lineno += self.line_offset - 1
        if hasattr(node, 'col_offset'):
            node.col_offset += self.col_offset
            node.end_col_offset += self.col_offset
        # Continue visiting child nodes
        return super().visit(node)


def stage_and_compile(func):
    fn_source = inspect.getsource(func)
    line_offset = inspect.getsourcelines(func)[1]

    d_fn_source = textwrap.dedent(fn_source)
    col_offset = len(fn_source.splitlines()[0]) - len(d_fn_source.splitlines()[0].lstrip())
    fn_source = d_fn_source
    # parse called function
    fn_ast = ast.parse(fn_source)
    LineNumberAdapter(line_offset, col_offset).visit(fn_ast)
    fn_ast.body[0] = stage_function(fn_ast.body[0], func.__globals__)
    # print(ast.dump(fn_ast))
    #print(ast.unparse(fn_ast))
    import hipy.lib.builtins
    globals = {**func.__globals__, "__builtins__": hipy.lib.builtins}
    exec(builtins.compile(fn_ast, filename=inspect.getsourcefile(func), mode="exec"), globals)
    compiled_fn = globals[func.__name__]
    return compiled_fn


def compile_function(hlc_function, arg_types, kw_types, module, fallback, debug):
    compiled_fn= hlc_function.get_compiled_fn()
    base_module=hlc_function.pyfunc.__module__
    mangled_func_name = compiled_fn.__name__  # todo: mangle_func_name(fn_node.name, [arg_types[n] for n in arg_order]) + "_".join(arg_order)
    # todo: check if function already exists
    # todo: real function signature
    problematic = []
    invalid_action_id = -1
    while True:
        # todo: fix
        module.block.ops.clear()
        fn = ir.Function(module, mangled_func_name, [t.ir_type() for t in arg_types], ir.void)
        ctxt = Context(module, fn.body, decisions=problematic, invalid_action_id=invalid_action_id, debug=debug, base_module=base_module)
        with ctxt.handle_action("args"):
            args = [ctxt.wrap(t.construct(v, ctxt)) for t, v in zip(arg_types, fn.args)]
        if not fallback:
            with ctxt.no_fallback():
                # todo: create abstract values for arguments
                res = compiled_fn(*args, _context=ctxt)
        else:
            res = compiled_fn(*args, _context=ctxt)
        if res is None:
            res = ir.Constant(fn.body, None, ir.void).result
        else:
            with ctxt.handle_action("func_res"):
                res = res.get_ir_value(ctxt)
        ir.Return(fn.body, [res])
        fn.res_type = res.type

        # print("events:")n
        # for event in ctxt.events:
        #    print(event)
        sys.stderr.flush()
        G = nx.DiGraph()
        ids = {}
        bad_node = -1
        G.add_node(bad_node)
        local_problematic = []
        reverse = []

        def get_id(val):
            loc_tuple = val.t_location
            if loc_tuple not in ids:
                ids[loc_tuple] = len(ids)
                reverse.append(val.t_location)
            return ids[loc_tuple]

        astart = time.time()
        for event in ctxt.events:
            # print(event)
            match event:
                case ValueAlias(val1=val1, val2=val2):
                    G.add_edge(get_id(val1), get_id(val2))
                    G.add_edge(get_id(val2), get_id(val1))
                case NestedValueAlias(nested=n, container=c):
                    G.add_edge(get_id(n), get_id(c))
                    G.add_edge(get_id(c), get_id(n))
                case ConvertedToPython(val=val):
                    G.add_edge(get_id(val), bad_node)
                case ValUsage(val=val):
                    if not G.has_node(get_id(val)):
                        G.add_node(get_id(val))
                    if nx.has_path(G, get_id(val), bad_node):
                        local_problematic.append(val.t_location)
                        # print("bad path", val.location)
                        shortest_path = nx.shortest_path(G, source=get_id(val), target=bad_node)
                        # print("shortest path",[ "bad" if n==bad_node else reverse[n] for n in shortest_path])
        # print("analyze",time.time()-astart)

        if len(local_problematic) == 0:
            return
        else:
            #print("problematic", local_problematic)
            # if invalid_action_id == -3:
            #    break
            #print(module)
            problematic.extend(local_problematic)
            invalid_action_id -= 1


def compile(fn, arg_types=None, fallback=False, debug=True):
    if arg_types is None:
        arg_types = []
    match fn:
        case HLCFunction():
            module = ir.Module()
            compile_function(fn, arg_types, {}, module, fallback, debug)

            return module
        case _:
            raise RuntimeError("can not compile function")
