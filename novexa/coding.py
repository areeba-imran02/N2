"""Static analysis only. User code is NEVER executed (no sandbox is available on Community Cloud)."""
import ast
RISKY = {"eval", "exec", "compile", "__import__", "open", "input"}
def analyze(code):
    if len(code) > 20000: raise ValueError("code too long (20k chars max)")
    try: tree = ast.parse(code)
    except SyntaxError as e: return {"ok": False, "syntax_error": f"line {e.lineno}: {e.msg}", "issues": [], "stats": {}}
    issues, imports, funcs, classes = [], set(), [], []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import): imports |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom): imports.add(n.module or "")
        elif isinstance(n, ast.FunctionDef):
            funcs.append(n.name)
            if not ast.get_docstring(n): issues.append(f"line {n.lineno}: function '{n.name}' has no docstring")
            for d in n.args.defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)): issues.append(f"line {n.lineno}: mutable default argument in '{n.name}'")
        elif isinstance(n, ast.ClassDef): classes.append(n.name)
        elif isinstance(n, ast.ExceptHandler) and n.type is None: issues.append(f"line {n.lineno}: bare 'except:' hides errors")
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in RISKY: issues.append(f"line {n.lineno}: use of '{n.func.id}' is risky")
    return {"ok": True, "issues": issues, "stats": {"lines": len(code.splitlines()), "functions": funcs, "classes": classes, "imports": sorted(imports)}}
