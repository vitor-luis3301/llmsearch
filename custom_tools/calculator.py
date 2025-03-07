import math, numexpr
from langchain_core.tools import StructuredTool

def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )

calc_tool = StructuredTool.from_function(
    name = "Calculator",
    description=(
        "A tool that allows for calculate expressions using Python's numexpr library. "
        "Usefull when user asks how to solve mathematical expressions"
    ),
    func=calculator
)