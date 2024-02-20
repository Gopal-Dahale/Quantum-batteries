from cudaq import spin
from functools import reduce

op_map = {
    'I': spin.i,
    'X': spin.x,
    'Y': spin.y,
    'Z': spin.z
}

def pauli_string_to_op(pauli_string):
    return reduce(lambda a,b: a*b, [op_map[op](q) for q, op in enumerate(pauli_string)])

def get_ham_from_dict(ham_dict):
    return reduce(lambda a, b: a + b, [coeff.real * pauli_string_to_op(pauli_string) for pauli_string, coeff in ham_dict.items()])

def rel_err(target, measured):
    return abs((target - measured) / target)


