# Contributing to Portfolio Risk Lab

## Commit Style

Use conventional commits:

| Prefix | When to use |
|---|---|
| `feat:` | new capability within the risk lab scope |
| `fix:` | bug correction |
| `test:` | adding or correcting tests |
| `refactor:` | structural change with no behavior change |
| `docs:` | documentation only |
| `chore:` | tooling, dependencies, CI |

For changes that touch a specific module, scope it: `feat(simulation):`, `fix(risk):`, etc.

Subject line: imperative mood, 72-character hard limit (enforced by commit-msg hook).
Body explains WHY, not WHAT. For significant changes, reference the relevant ADR or
architecture section: `Architecture ref: §10, ADR-003`.

## Before Contributing

- Read `portfolio_risk_lab_c4_architecture.md` — scope boundaries are strict.
- The product is a **quantitative model-risk analysis tool**. Changes that introduce
  portfolio optimization, efficient frontier construction, live trading, broker
  integration, a database, or distributed infrastructure are out of scope.

## Development Setup

```bash
uv sync --extra dev
git config core.hooksPath .githooks
```

## Quality Gates

All of the following must pass before a commit lands:

| Check | Command |
|---|---|
| Format | `uv run ruff format --check .` |
| Lint | `uv run ruff check .` |
| Type check | `uv run mypy src` |
| Tests | `uv run pytest -m "not slow"` |

The pre-commit hook runs these automatically.
Slow statistical sanity checks (`-m slow`) run on demand: `uv run pytest -m slow`.

## Testing Requirements

| Change type | Required tests |
|---|---|
| New simulation model | Shape test, reproducibility under fixed seed, statistical sanity (slow) |
| New risk metric | Deterministic fixture with analytically known expected value |
| New portfolio validation rule | Test for valid and invalid input |
| New scenario | Deterministic output test with explicit expected loss |
| New diagnostic finding | Test that finding is generated from known metric differences and suppressed when models agree |

## Methodological Boundary

Portfolio Risk Lab compares risk conclusions across models.
Changes that imply one model is universally correct, blend model outputs into
a single number, or introduce live/production capabilities are out of scope.

All stochastic procedures must accept and record a `random_seed`.
