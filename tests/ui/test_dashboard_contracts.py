from __future__ import annotations

import ast
from io import StringIO
from pathlib import Path

import pytest

from app.validation import AnalysisParameterValidator, PortfolioUploadValidator


def test_portfolio_csv_upload_rejects_missing_required_columns() -> None:
    csv_file = StringIO("ticker\nAAPL\nMSFT\n")

    with pytest.raises(ValueError, match="weight"):
        PortfolioUploadValidator().validate(csv_file)


@pytest.mark.parametrize("confidence_level", [0.0, -0.1, 1.0, 1.2])
def test_analysis_parameter_form_rejects_confidence_level_outside_open_unit_interval(
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        AnalysisParameterValidator().validate(
            confidence_level=confidence_level,
            horizon=10,
            n_simulations=1_000,
            loss_threshold=0.10,
            random_seed=42,
        )


@pytest.mark.parametrize("horizon", [0, -1])
def test_analysis_parameter_form_rejects_non_positive_horizon(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        AnalysisParameterValidator().validate(
            confidence_level=0.95,
            horizon=horizon,
            n_simulations=1_000,
            loss_threshold=0.10,
            random_seed=42,
        )


def test_dashboard_does_not_import_analytical_modules_directly() -> None:
    prohibited_modules = (
        "src.simulation",
        "src.risk",
        "src.returns",
        "src.scenarios",
    )

    for python_file in Path("app").glob("*.py"):
        tree = ast.parse(python_file.read_text(), filename=str(python_file))
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

        assert not any(
            module == prohibited or module.startswith(f"{prohibited}.")
            for module in imported_modules
            for prohibited in prohibited_modules
        ), f"{python_file} imports an analytical module directly"
