"""A custom tool: exact arithmetic evaluation.

LLMs predict tokens, not compute results - they're unreliable at exact
multi-digit arithmetic. This tool lets the model delegate any math to
real code instead of guessing at an answer.
"""

import ast
import operator

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": (
            "Evaluate a basic arithmetic expression and return the exact "
            "numeric result. Use this for any math instead of computing it "
            "yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": 'An arithmetic expression using +, -, *, /, ** and parentheses, e.g. "2 * 5"',
                }
            },
            "required": ["expression"],
        },
    },
}

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression - parses to an AST and
    walks it directly rather than calling eval(), so arbitrary code
    from a model-supplied string can never execute."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"error: could not evaluate {expression!r}: {exc}"
