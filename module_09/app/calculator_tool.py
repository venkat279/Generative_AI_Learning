"""A custom tool: exact arithmetic evaluation, as a LangChain tool.

Same idea and safety approach as Module 7's calculator_tool.py (an
AST-based evaluator, not eval() - a tool the model can invoke is a
real injection surface), reimplemented with LangChain's @tool
decorator instead of a raw Gemini tool-call schema, since this
module's agent uses LangChain/LangGraph's own tool-calling mechanism.
"""

import ast
import operator

from langchain_core.tools import tool

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


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the exact numeric result. Use this for any math instead of computing it yourself."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"error: could not evaluate {expression!r}: {exc}"
