import re
import inspect
from functools import wraps
from typing import get_type_hints, Callable, Any

DOCSTRING_PLACEHOLDER_REGEX = re.compile(r'\{(\w+)\}')

def templated_docstring(template: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    A decorator that adds a templated docstring to a function.
    The template can include placeholders for function name, parameters, and return type.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Get return type hint if available
        return_type = type_hints.get('return', Any).__name__
        
        # Format parameters with type hints
        params = []
        for name, param in sig.parameters.items():
            if name != 'self':  # Skip self parameter for methods
                type_hint = type_hints.get(name, Any).__name__
                params.append(f"{name}: {type_hint}")
        
        # Replace placeholders in template
        docstring = template.format(
            func_name=func.__name__,
            params="\n".join(params),
            return_type=return_type
        )
        
        func.__doc__ = docstring
        return func
    
    return decorator
