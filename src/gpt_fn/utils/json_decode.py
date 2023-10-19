
from functools import wraps

def state(fn):
    @wraps(fn)
    def wrapper(input: str, stack: list[str] = []) -> str:
        r = fn(input, stack)
        if r is None:
            raise ValueError("Invalid JSON {input} in {fn.__name__}")
        return r
    return wrapper

def state_start(input: str) -> str:
    return state_root_object(input, ["$"])

@state
def state_root_object(input: str, stack: list[str]) -> str:
    input = input.strip()

    if input[0] == "{":
        return input[0] + state_object(input[1:], stack + ["{"])
    
@state
def state_finish(input: str, stack: list[str]) -> str:
    input = input.strip()

    if not input:
        return ""

@state
def state_object(input: str, stack: list[str]) -> str:
    input = input.strip()

    if input[0] == "}" and stack[-1] == "{":
        return input[0] + state_post_object(input[1:], stack[:-1])

    if input[0] == '"':
        return input[0] + state_property_string(input[1:], stack)

@state
def state_post_object(input: str, stack: list[str]) -> str:
    if stack[-1] in {"{", "["}:
        return state_post_value(input, stack)

    if stack[-1] == "$":
        return state_finish(input, stack)
    
@state
def state_post_property(input: str, stack: list[str]) -> str:
    input = input.strip()

    if input[0] == ":":
        return input[0] + state_value(input[1:], stack)

@state
def state_value(input: str, stack: list[str]) -> str:
    input = input.strip()

    if input[0] == "f":
        assert input[:5] == "false"
        return input[:5] + state_post_value(input[5:], stack)
    if input[0] == "t":
        assert input[:4] == "true"
        return input[:4] + state_post_value(input[4:], stack)
    if input[0] == "n":
        assert input[:4] == "null"
        return input[:4] + state_post_value(input[4:], stack)
    if input[0].isdigit() or input[0] == "-":
        return input[0] + state_int(input, stack)
    if input[0] == '"':
        return input[0] + state_value_string(input[1:], stack)
    if input[0] == "{":
        return input[0] + state_object(input[1:], stack + ["{"])
    if input[0] == "[":
        return input[0] + state_value(input[1:], stack + ["["]) 
    if input[0] == "]":
        assert stack[-1] == "["
        return input[0] + state_post_value(input[1:], stack[:-1])
    
@state
def state_post_value(input: str, stack: list[str]) -> str:
    input = input.strip()

    if input[0] == ",":
        if stack[-1] == "[":
            return input[0] + state_value(input[1:], stack)
        if stack[-1] == "{":
            return input[0] + state_object(input[1:], stack)
        return
    elif input[0] == "]":
        assert stack[-1] == "["
        return input[0] + state_post_value(input[1:], stack[:-1])
    elif input[0] == "}":
        assert stack[-1] == "{"
        return input[0] + state_post_object(input[1:], stack[:-1])

@state
def state_value_string(input: str, stack: list[str]) -> str:
    if input[0] == '"':
        # TODO: 
        return input[0] + state_post_value(input[1:], stack)

    if input[0] == "\\":
        return input[0] + state_value_string(input[1:], stack + ["v"])
    
    return input[0] + state_value_string(input[1:], stack)
    
def state_property_string(input: str, stack: list[str]) -> str:
    if input[0] == '"':
        return input[0] + state_post_property(input[1:], stack)

    if input[0] == "\\":
        return input[0] + state_property_string(input[1:], stack + ["p"])
    
    return input[0] + state_property_string(input[1:], stack)

@state
def state_escape_char(input: str, stack: list[str]) -> str:
    if input[0] == "u":
        int(input[1:5], 16)
    
        if stack[-1] == "v":
            return input[0:5] + state_value_string(input[5:], stack[:-1]) 
        if stack[-1] == "p":
            return input[0] + state_property_string(input[5:], stack[:-1])

        return        
    if input[0] in {"\\", "/", '"', "b", "f", "n", "r", "t"}:
        if stack[-1] == "v":
            return input[0] + state_value_string(input[1:], stack[:-1])
        if stack[-1] == "p":
            return input[0] + state_property_string(input[1:], stack[:-1])
        return

@state
def state_int(input: str, stack: list[str]) -> str:
    if input[0].isdigit():
        return input[0] + state_int(input[1:], stack)
    if input[0] == ".":
        return input[0] + state_double(input[1:], stack)
    if input[0] == ",":
        if stack[-1] == "[":
            return input[0] + state_value(input[1:], stack)
        if stack[-1] == "{":
            return input[0] + state_object(input[1:], stack)
        return
    if input[0] == "}":
        assert stack[-1] == "{"
        return input[0] + state_post_int_parent(input[1:], stack)
    if input[0] == "]":
        assert stack[-1] == "["
        return input[0] + state_post_int_parent(input[1:], stack)
    if input[0].isspace():
        return input[0] + state_post_value(input[1:], stack)
    
@state
def state_post_int_parent(input: str, stack: list[str]) -> str:
    if stack[-1] == "[":
        return state_post_value(input, stack[:-1])
    if stack[-1] == "{":
        return state_post_object(input, stack[:-1])
    
@state
def state_double(input: str, stack: list[str]) -> str:
    if input[0].isdigit():
        return input[0] + state_double(input[1:], stack)
    if input[0] == ",":
        if stack[-1] == "[":
            return input[0] + state_value(input[1:], stack)
        if stack[-1] == "{":
            return input[0] + state_object(input[1:], stack)
        return
    if input[0] == "}":
        assert stack[-1] == "{"
        return input[0] + state_post_int_parent(input[1:], stack)
    if input[0] == "]":
        assert stack[-1] == "["
        return input[0] + state_post_int_parent(input[1:], stack)
    if input[0].isspace():
        return input[0] + state_post_value(input[1:], stack)
    if input[0] == {"e", "E"}:
        return input[0] + state_exponent_sign(input[1:], stack)
    
@state
def state_exponent_sign(input: str, stack: list[str]) -> str:
    if input[0] in {"+", "-"}:
        return input[0] + state_exponent_digits(input[1:], stack)

@state
def state_exponent_digits(input: str, stack: list[str]) -> str:
    if input[0].isdigit():
        return input[0] + state_exponent_digits(input[1:], stack)
    if input[0] == ",":
        if stack[-1] == "[":
            return input[0] + state_value(input[1:], stack)
        if stack[-1] == "{":
            return input[0] + state_object(input[1:], stack)
        return
    if input[0] == "}":
        assert stack[-1] == "{"
        return input[0] + state_post_int_parent(input[1:], stack)
    if input[0] == "]":
        assert stack[-1] == "["
        return input[0] + state_post_int_parent(input[1:], stack)
    if input[0].isspace():
        return input[0] + state_post_value(input[1:], stack)
    

def correct_json(json_str: str) -> str:
    return state_start(json_str)