"""An AST-restricted Python evaluator for the analyst.

Code is parsed and checked against an allowlist before anything executes.
Imports, dunder attribute access, ``eval``/``exec``/``open`` and unbounded
iteration are rejected. Loops and comprehensions carry an injected counter,
``range`` is length-capped, and ``**`` and ``*`` are routed through guards that
reject results which would exhaust memory. Execution runs in a worker thread
under a wall-clock deadline.
"""

from __future__ import annotations

import ast
import itertools
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import SandboxViolation

_BANNED_CALLS = frozenset(
    """eval exec compile open input globals locals vars getattr setattr delattr __import__
    breakpoint memoryview bytearray classmethod staticmethod super type object""".split()
)

MAX_NODES = 900
MAX_LOOPS = 200_000
MAX_RANGE = 1_000_000
MAX_SEQUENCE = 1_000_000
MAX_POW_DIGITS = 24
DEFAULT_TIMEOUT = 2.5

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.keyword,
    ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript, ast.Slice, ast.Index if hasattr(ast, "Index") else ast.Slice,
    ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass, ast.Return,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.And, ast.Or, ast.Not, ast.In, ast.NotIn,
    ast.IfExp, ast.Attribute, ast.Starred, ast.JoinedStr, ast.FormattedValue,
)

_ALLOWED_ATTRS = frozenset(
    ["append", "extend", "insert", "pop", "index", "count", "sort", "reverse", "keys", "values", "items", "get", "update", "lower", "upper", "strip", "split", "join", "replace", "startswith", "endswith", "format", "title", "real", "imag", "numerator", "denominator", "copy", "add", "discard"]
)

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len, "round": round, "sorted": sorted,
    "range": range, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "any": any,
    "all": all, "int": int, "float": float, "str": str, "bool": bool, "list": list, "tuple": tuple,
    "dict": dict, "set": set, "reversed": reversed, "divmod": divmod, "pow": pow, "print": lambda *a, **k: None,
}

_MATH = {k: getattr(math, k) for k in ("sqrt", "log", "log2", "log10", "exp", "floor", "ceil", "fabs", "pi", "e", "inf", "isfinite", "hypot", "sin", "cos", "tan")}
_STATS = {
    "mean": statistics.mean, "median": statistics.median, "stdev": lambda x: statistics.stdev(x) if len(x) > 1 else 0.0,
    "pstdev": statistics.pstdev, "variance": lambda x: statistics.variance(x) if len(x) > 1 else 0.0,
    "mode": statistics.mode,
}


def _extra_helpers() -> dict[str, Any]:
    def pct_change(a: float, b: float) -> float:
        return (b - a) / a * 100.0 if a else float("inf")

    def cagr(first: float, last: float, periods: float) -> float:
        if first <= 0 or periods <= 0:
            return float("nan")
        return ((last / first) ** (1.0 / periods) - 1.0) * 100.0

    def growth_series(values: list[float]) -> list[float]:
        return [round(pct_change(a, b), 4) for a, b in itertools.pairwise(values)]

    return {"pct_change": pct_change, "cagr": cagr, "growth_series": growth_series}


@dataclass(slots=True)
class ReplResult:
    ok: bool
    value: Any = None
    stdout: str = ""
    error: str = ""
    elapsed_ms: float = 0.0
    variables: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "value": self.value,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "variables": self.variables,
        }


def validate(code: str) -> ast.Module:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"syntax error: {exc.msg} (line {exc.lineno})") from exc
    for count, node in enumerate(ast.walk(tree), start=1):
        if count > MAX_NODES:
            raise SandboxViolation("program too large")
        if not isinstance(node, _ALLOWED_NODES):
            raise SandboxViolation(f"construct not allowed: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr not in _ALLOWED_ATTRS
        ):
            raise SandboxViolation(f"attribute not allowed: .{node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SandboxViolation(f"name not allowed: {node.id}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _BANNED_CALLS
        ):
            raise SandboxViolation(f"call not allowed: {node.func.id}()")
    return tree


class _GuardedRange:
    """``range`` with a hard length cap and a per-item tick."""

    __slots__ = ("_r", "_tick")

    def __init__(self, tick: Any, *args: int) -> None:
        r = range(*args)
        if len(r) > MAX_RANGE:
            raise SandboxViolation(f"range too large ({len(r)} > {MAX_RANGE})")
        self._r, self._tick = r, tick

    def __iter__(self) -> Any:
        for x in self._r:
            self._tick()
            yield x

    def __len__(self) -> int:
        return len(self._r)

    def __contains__(self, x: object) -> bool:
        return x in self._r

    def __getitem__(self, i: int) -> Any:
        return self._r[i]


def _safe_pow(a: Any, b: Any) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(b) > 512:
            raise SandboxViolation("exponent too large")
        if a not in (0, 1, -1) and abs(b) * math.log10(max(abs(a), 1.0000001)) > MAX_POW_DIGITS:
            raise SandboxViolation("power result too large")
    return a**b


def _safe_mul(a: Any, b: Any) -> Any:
    for x, y in ((a, b), (b, a)):
        if isinstance(x, (str, list, tuple, bytes)) and isinstance(y, int) and len(x) * max(y, 0) > MAX_SEQUENCE:
            raise SandboxViolation("sequence repetition too large")
    return a * b


def _build_env(variables: dict[str, Any] | None) -> dict[str, Any]:
    env: dict[str, Any] = {"__builtins__": {}}
    env.update(_SAFE_BUILTINS)
    env.update(_MATH)
    env.update(_STATS)
    env.update(_extra_helpers())
    for k, v in (variables or {}).items():
        if k.isidentifier() and not k.startswith("_"):
            env[k] = v
    return env


def run_sync(code: str, variables: dict[str, Any] | None = None, *, timeout: float = DEFAULT_TIMEOUT) -> ReplResult:
    tree = validate(code)
    env = _build_env(variables)
    t0 = time.perf_counter()

    # Bound loops by injecting a counter guard into every loop body.
    guard = {"n": 0}

    def _tick() -> None:
        guard["n"] += 1
        if guard["n"] > MAX_LOOPS:
            raise SandboxViolation("iteration limit exceeded")
        if time.perf_counter() - t0 > timeout:
            raise TimeoutError("execution timed out")

    def _tickt() -> bool:
        _tick()
        return True

    env["__tick__"] = _tick
    env["__tickt__"] = _tickt
    env["__pow__"] = _safe_pow
    env["__mul__"] = _safe_mul
    env["range"] = lambda *a: _GuardedRange(_tick, *a)
    tree = _inject_ticks(tree)
    ast.fix_missing_locations(tree)

    # Report the last expression's value, mirroring notebook semantics.
    last_expr: ast.Expr | None = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body.pop()  # type: ignore[assignment]

    try:
        exec(compile(tree, "<analyst>", "exec"), env)
        value = None
        if last_expr is not None:
            value = eval(compile(ast.Expression(last_expr.value), "<analyst>", "eval"), env)
    except SandboxViolation:
        raise
    except TimeoutError as exc:
        return ReplResult(ok=False, error=str(exc), elapsed_ms=(time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return ReplResult(
            ok=False, error=f"{type(exc).__name__}: {exc}", elapsed_ms=(time.perf_counter() - t0) * 1000
        )

    exported = {
        k: v
        for k, v in env.items()
        if not k.startswith("_") and k not in _SAFE_BUILTINS and k not in _MATH and k not in _STATS
        and k != "range"
        and isinstance(v, (int, float, str, bool, list, tuple, dict))
    }
    return ReplResult(
        ok=True,
        value=_jsonable(value),
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        variables={k: _jsonable(v) for k, v in list(exported.items())[:24]},
    )


def _inject_ticks(tree: ast.Module) -> ast.Module:
    """Rewrite loops, comprehensions and explosive operators with guards."""

    def tick_stmt() -> ast.Expr:
        return ast.Expr(value=ast.Call(func=ast.Name(id="__tick__", ctx=ast.Load()), args=[], keywords=[]))

    def tick_test() -> ast.Call:
        return ast.Call(func=ast.Name(id="__tickt__", ctx=ast.Load()), args=[], keywords=[])

    class Injector(ast.NodeTransformer):
        def visit_For(self, node: ast.For) -> ast.AST:
            self.generic_visit(node)
            node.body = [ast.copy_location(tick_stmt(), node), *node.body]
            return node

        def visit_While(self, node: ast.While) -> ast.AST:
            self.generic_visit(node)
            node.body = [ast.copy_location(tick_stmt(), node), *node.body]
            return node

        def visit_comprehension(self, node: ast.comprehension) -> ast.AST:
            self.generic_visit(node)
            node.ifs = [*node.ifs, ast.copy_location(tick_test(), node.iter)]
            return node

        def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
            self.generic_visit(node)
            fn = {ast.Pow: "__pow__", ast.Mult: "__mul__"}.get(type(node.op))
            if fn is None:
                return node
            return ast.copy_location(
                ast.Call(func=ast.Name(id=fn, ctx=ast.Load()), args=[node.left, node.right], keywords=[]), node
            )

    return Injector().visit(tree)


def _jsonable(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in list(v)[:200]]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in list(v.items())[:200]}
    return str(v)[:300]


async def run(code: str, variables: dict[str, Any] | None = None, *, timeout: float = DEFAULT_TIMEOUT) -> ReplResult:
    from app.core.concurrency import run_blocking

    try:
        return await run_blocking(run_sync, code, variables, timeout=timeout)
    except SandboxViolation as exc:
        return ReplResult(ok=False, error=f"sandbox: {exc.message}")
