"""Agent tools as plain Python functions. Each takes one string and returns a string."""
import ast
import operator
from datetime import datetime
from retrieve import retrieve_context

def search_docs(query: str, mode: str | None = None) -> str:
    """Hybrid RAG over the company documents (benefits, handbook, Zava, market report)."""
    return retrieve_context(query, mode=mode)

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}

def calculator(expression: str) -> str:
    """Evaluate arithmetic safely (no eval): + - * / ** % // and parentheses."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    try:
        expr = expression.replace("^", "**").replace(",", "").replace("x", "*")
        return str(round(ev(ast.parse(expr, mode="eval").body), 6))
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"

def web_search(query: str) -> str:
    """DuckDuckGo search, top 5 results."""
    from ddgs import DDGS
    try:
        results = DDGS().text(query, max_results=5)
    except Exception as e:
        return f"Error: web search failed ({e})"
    return "\n---\n".join(f"{r['title']} ({r['href']})\n{r['body']}" for r in results) or "No results."

def get_datetime(_: str = "") -> str:
    """Current local date and time."""
    return datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")

TOOLS = {"search_docs": search_docs, "calculator": calculator, "web_search": web_search, "get_datetime": get_datetime}

if __name__ == "__main__":
    assert calculator("(2+3)*4") == "20"
    assert calculator("2^10") == "1024"
    assert calculator("1,200 * 0.15") == "180.0"
    assert calculator("__import__('os')").startswith("Error")
    print(get_datetime())
    print(web_search("LangGraph")[:300])
    print("tools ok")
