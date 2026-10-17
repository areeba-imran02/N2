"""Safe scientific calculator: whitelisted AST evaluator, no eval/exec."""
import ast, math, operator as op

class CalcError(ValueError): pass
_BIN = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.FloorDiv: op.floordiv}
_CONST = {"pi": math.pi, "e": math.e}
_TRIG = {"sin": math.sin, "cos": math.cos, "tan": math.tan}
_ITRIG = {"asin": math.asin, "acos": math.acos, "atan": math.atan}
_FUNCS = {"sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x), "ln": math.log,
          "log": math.log10, "log10": math.log10, "abs": abs, "exp": math.exp}

def _fact(n):
    if n != int(n) or n < 0 or n > 170: raise CalcError("factorial needs an integer in 0..170")
    return math.factorial(int(n))

def evaluate(expr, mode="RAD"):
    """Returns (value, trace). Every trace event is appended only when that operation really ran."""
    if mode not in ("DEG", "RAD"): raise CalcError("mode must be DEG or RAD")
    trace = [("received", expr)]
    src = expr.replace("^", "**").replace("°", "*pi/180").replace("×", "*").replace("÷", "/")
    if len(src) > 300: raise CalcError("expression too long")
    try: tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e: raise CalcError(f"invalid syntax: {e.msg}")
    trace.append(("parsed", ast.dump(tree.body)[:120]))
    trace.append(("angle_mode", mode))

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): return n.value
        if isinstance(n, ast.Name):
            if n.id in _CONST: return _CONST[n.id]
            raise CalcError(f"unknown name '{n.id}'")
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand); return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 1000: raise CalcError("exponent too large")
            try: r = _BIN[type(n.op)](a, b)
            except ZeroDivisionError: raise CalcError("division by zero")
            except OverflowError: raise CalcError("overflow")
            if isinstance(r, complex): raise CalcError("complex result not supported")
            trace.append(("computed", f"{a!r} {type(n.op).__name__} {b!r} = {r!r}")); return r
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and not n.keywords:
            f = n.func.id; args = [ev(a) for a in n.args]
            if len(args) != 1: raise CalcError(f"{f} takes one argument")
            x = args[0]
            try:
                if f in _TRIG:
                    r = _TRIG[f](math.radians(x) if mode == "DEG" else x)
                    if abs(r) < 1e-15: r = 0.0
                elif f in _ITRIG: r = _ITRIG[f](x); r = math.degrees(r) if mode == "DEG" else r
                elif f == "fact": r = _fact(x)
                elif f in _FUNCS: r = _FUNCS[f](x)
                else: raise CalcError(f"unknown function '{f}'")
            except CalcError: raise
            except (ValueError, OverflowError): raise CalcError(f"domain error in {f}({x})")
            trace.append(("computed", f"{f}({x!r}) = {r!r}")); return r
        raise CalcError("unsupported expression element")
    val = ev(tree.body)
    if isinstance(val, float) and not math.isfinite(val): raise CalcError("non-finite result")
    trace.append(("result", repr(val)))
    return val, trace

def percent(base, pct): return base * pct / 100.0
def fmt(v, digits=12): return str(v) if isinstance(v, int) and abs(v) < 10**15 else f"{v:.{digits}g}"

# ---- symbolic + matrix (sympy), input validated by AST whitelist before parsing ----
import json as _json
import sympy as _sp
_SYM_OK = {"sin", "cos", "tan", "asin", "acos", "atan", "sqrt", "ln", "log", "exp", "abs"}
def _check_sym(expr, allowed_vars):
    if len(expr) > 200: raise CalcError("expression too long")
    if expr.count("=") > 1: raise CalcError("only one '=' allowed")
    for side in expr.split("="): _check_sym(side or "0", allowed_vars) if "=" in expr else None
    if "=" in expr: return expr.replace("^", "**")
    try: t = ast.parse(expr.replace("^", "**"), mode="eval")
    except SyntaxError as e: raise CalcError(f"invalid syntax: {e.msg}")
    for n in ast.walk(t):
        if isinstance(n, ast.Name) and n.id not in _SYM_OK | {"pi", "e"} | set(allowed_vars): raise CalcError(f"name '{n.id}' not allowed")
        if isinstance(n, (ast.Attribute, ast.Subscript, ast.Lambda, ast.Dict, ast.List, ast.Starred)): raise CalcError("unsupported syntax")
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id in _SYM_OK): raise CalcError("function not allowed")
    return expr.replace("^", "**")

def symbolic(op_name, expr, var="x"):
    """op_name: simplify | diff | integrate | solve. Returns (result_str, steps)."""
    if not var.isidentifier() or len(var) > 3: raise CalcError("bad variable")
    src = _check_sym(expr, [var]); x = _sp.Symbol(var)
    loc = {var: x, "ln": _sp.log, "log": lambda v: _sp.log(v, 10), "e": _sp.E, "pi": _sp.pi}
    try:
        f = _sp.parse_expr(src.split('=')[0] if op_name == 'solve' else src, local_dict=loc)
        if op_name == "simplify": r = _sp.simplify(f)
        elif op_name == "diff": r = _sp.diff(f, x)
        elif op_name == "integrate": r = _sp.integrate(f, x)
        elif op_name == "solve":
            lhs, _, rhs = src.partition("=")
            r = _sp.solve(_sp.parse_expr(lhs, local_dict=loc) - (_sp.parse_expr(rhs, local_dict=loc) if rhs else 0), x)
        else: raise CalcError("unknown operation")
    except CalcError: raise
    except Exception as e: raise CalcError(f"symbolic error: {type(e).__name__}")
    steps = [f"Parsed input as {f}", f"Applied sympy.{op_name} with respect to {var}", f"Result: {r}"]
    return str(r), steps

def matrix_op(op_name, a, b=None):
    """a, b: JSON strings or nested lists. op_name: det | inv | transpose | mul | add | eigvals."""
    try:
        A = _sp.Matrix(_json.loads(a) if isinstance(a, str) else a)
        B = _sp.Matrix(_json.loads(b) if isinstance(b, str) else b) if b not in (None, "") else None
    except Exception: raise CalcError("matrix must be JSON like [[1,2],[3,4]]")
    if A.rows * A.cols > 100: raise CalcError("matrix too large")
    try:
        if op_name == "det": return str(A.det())
        if op_name == "transpose": return str(A.T.tolist())
        if op_name == "inv":
            if A.det() == 0: raise CalcError("matrix is singular")
            return str(A.inv().tolist())
        if op_name == "eigvals": return str(A.eigenvals())
        if B is None: raise CalcError("second matrix required")
        if op_name == "mul": return str((A * B).tolist())
        if op_name == "add": return str((A + B).tolist())
    except _sp.ShapeError: raise CalcError("incompatible matrix shapes")
    except _sp.NonSquareMatrixError: raise CalcError("matrix must be square")
    raise CalcError("unknown matrix operation")
