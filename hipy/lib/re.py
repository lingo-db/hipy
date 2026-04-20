import re as _re
import sys
import hipy
from hipy import intrinsics, ir
from hipy.value import Value, Type, SimpleType, static_object, ValueHolder, CValue
from hipy.value import raw_module

__HIPY_MODULE__ = "re"

# Import original re module for reference  
original = raw_module(_re)
hipy.register(sys.modules[__name__])


@hipy.classdef
class Match(static_object["string", "_groups","_numGroups", "_re","_method", "_flags"]):
    """Represents a regex match result"""
    
    def __init__(self, string,groups, numGroups, re_obj, method, flags):
        super().__init__(lambda args: Match(*args), string, groups, numGroups, re_obj, method, flags)
    @staticmethod
    @hipy.raw
    def __create__(string, groups, numGroups, re_obj, method, flags, _context):
        return hipy.value.ValueHolder(Match(string, groups, numGroups, re_obj, method, flags),_context)
    @hipy.compiled_function
    def __topython__(self):
        return original.search(self._re, self.string, self._flags)

    @hipy.compiled_function
    def __str__(self):
        return "<Match object; span="+str(self.span())+", match='"+str(self.group())+"'>"

    @hipy.compiled_function
    def group(self, group_num=0):
        start, end = self._groups[group_num]
        return self.string[start:end]
    @hipy.compiled_function
    def span(self, group_num=0):
        return self._groups[group_num]
@hipy.raw
def _is_simple_regex(pattern, _context):
    match pattern.value:
        case CValue(cval=const_pattern) if isinstance(const_pattern, str):


            # Support simple patterns like:
            # - Literal strings with parentheses for groups: "(\\d+)_zpid/$"
            # - Basic character classes: \\d, \\w, \\s
            # - Basic quantifiers: +, *, ?
            # - Anchors: ^, $

            # For now, only support very basic patterns
            simple_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()[]{}+*?^$._-/:")
            escape_sequences = ["\\d", "\\w", "\\s", "\\D", "\\W", "\\S"]

            i = 0
            while i < len(const_pattern):
                if const_pattern[i] == '\\' and i + 1 < len(const_pattern):
                    if const_pattern[i:i+2] not in escape_sequences:
                        return _context.constant(False, 0)
                    i += 2
                elif const_pattern[i] in simple_chars:
                    i += 1
                else:
                    return _context.constant(False, 0)

            # Check for unsupported features
            unsupported = ["|", "(?", "(?:", "(?=", "(?!", "(?<=", "(?<!"]
            for feature in unsupported:
                if feature in const_pattern:
                    return _context.constant(False, 0)

            return _context.constant(True, 0)
        case _:
            return _context.constant(False, 0)

@hipy.raw
def _count_groups(pattern, _context):
    match pattern.value:
        case CValue(cval=const_pattern) if isinstance(const_pattern, str):
            # Count the number of capturing groups in the pattern
            count = 0
            escape = False
            in_char_class = False
            for char in const_pattern:
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                elif char == '[':
                    in_char_class = True
                elif char == ']':
                    in_char_class = False
                elif char == '(' and not in_char_class:
                    count += 1
            return _context.constant(count, 0)
        case _:
            raise NotImplementedError()

@hipy.raw
def _const_sized_tuple(list, constNum, _context):
    match constNum.value:
        case CValue(cval=n) if isinstance(n, int):

            elems = []
            for i in range(n):
                elems.append( _context.get_item(list, _context.constant(i, i),i))
            return _context.create_tuple(elems)
        case _:
            intrinsics.not_implemented()



@hipy.compiled_function
def search(pattern, string, flags=0):
   if _is_simple_regex(pattern):
       numGroups = _count_groups(pattern)
       res_type= intrinsics.create_type(list, intrinsics.create_type(tuple,[int,int]), numGroups+1)

       match_res = intrinsics.call_builtin("regex.search", res_type, [pattern, string])
       hasMatch = len(match_res) > 0
       if hasMatch:
           groupsList = match_res
           groupsTuple = _const_sized_tuple(groupsList, numGroups+1)
           return _MaybeNone(False, Match(string, groupsTuple, numGroups+1, pattern, "search", flags))
       else:
           groupsTuple = _const_sized_tuple([(0,0)] * (numGroups+1), numGroups+1)
           return _MaybeNone(True, Match(string, groupsTuple, numGroups+1, pattern, "search", flags))
   else:
       intrinsics.not_implemented()