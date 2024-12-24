from __future__ import annotations

import ast
import copy
from types import ModuleType
from typing import Any, Optional, Union

import gradio as gr

import modules.scripts as scripts
from modules.processing import Processed
from modules.shared import cmd_opts


def convert_expr_to_expression(expr: ast.Expr) -> ast.Expression:
    """Convert an ast.Expr node to ast.Expression for evaluation.
    
    Args:
        expr: The expression node to convert
        
    Returns:
        An ast.Expression node ready for compilation
    """
    expr.lineno = 0
    expr.col_offset = 0
    result = ast.Expression(
        body=expr.value,
        lineno=0,
        col_offset=0
    )
    return result


def exec_with_return(code: str, module: ModuleType) -> Any:
    """Execute code and return its value if the last statement is an expression.
    
    Args:
        code: The Python code to execute
        module: The module context for execution
        
    Returns:
        The result of the last expression if applicable, None otherwise
        
    Reference:
        https://stackoverflow.com/a/52361938/5862977
    """
    code_ast = ast.parse(code)

    # Split the AST into initialization and last statement
    init_ast = copy.deepcopy(code_ast)
    init_ast.body = code_ast.body[:-1]

    last_ast = copy.deepcopy(code_ast)
    last_ast.body = code_ast.body[-1:]

    # Execute all but the last statement
    exec(compile(init_ast, "<ast>", "exec"), module.__dict__)
    
    # Handle the last statement
    if isinstance(last_ast.body[0], ast.Expr):
        return eval(
            compile(convert_expr_to_expression(last_ast.body[0]), "<ast>", "eval"),
            module.__dict__
        )
    else:
        exec(compile(last_ast, "<ast>", "exec"), module.__dict__)


class Script(scripts.Script):
    """Custom code execution script for image processing."""

    def title(self) -> str:
        return "Custom code"

    def show(self, is_img2img: bool) -> bool:
        return cmd_opts.allow_code

    def ui(self, is_img2img: bool) -> list[Union[gr.Code, gr.Number]]:
        example = """from modules.processing import process_images

p.width = 768
p.height = 768
p.batch_size = 2
p.steps = 10

return process_images(p)
"""

        code = gr.Code(
            value=example,
            language="python",
            label="Python code",
            elem_id=self.elem_id("code")
        )
        
        indent_level = gr.Number(
            label='Indent level',
            value=2,
            precision=0,
            elem_id=self.elem_id("indent_level")
        )

        return [code, indent_level]

    def run(
        self,
        p: Any,
        code: str,
        indent_level: int
    ) -> Processed:
        """Execute custom code and return processed results.
        
        Args:
            p: Processing parameters
            code: Python code to execute
            indent_level: Number of spaces for code indentation
            
        Returns:
            Processed result object
            
        Raises:
            AssertionError: If --allow-code option is not enabled
        """
        assert cmd_opts.allow_code, '--allow-code option must be enabled'

        display_result_data: list[Any] = [[], -1, ""]

        def display(
            imgs: list[Any],
            s: int = display_result_data[1],
            i: str = display_result_data[2]
        ) -> None:
            display_result_data[0] = imgs
            display_result_data[1] = s
            display_result_data[2] = i

        # Create isolated module for code execution
        module = ModuleType("testmodule")
        module.__dict__.update(globals())
        module.p = p
        module.display = display

        # Prepare code with proper indentation
        indent = " " * indent_level
        indented = code.replace('\n', f"\n{indent}")
        body = f"""def __webuitemp__():
{indent}{indented}
__webuitemp__()"""

        result = exec_with_return(body, module)

        if isinstance(result, Processed):
            return result

        return Processed(p, *display_result_data)