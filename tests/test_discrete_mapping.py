import ast
import types
import numpy as np
from pathlib import Path

# Dynamically load only the discrete_mapping function from train.py
train_path = Path(__file__).resolve().parents[1] / "train.py"
source = train_path.read_text(encoding="utf-8")
module_ast = ast.parse(source)

# Extract the discrete_mapping function definition
func_node = next(
    node for node in module_ast.body if isinstance(node, ast.FunctionDef) and node.name == "discrete_mapping"
)

module = types.ModuleType("train_discrete")
exec(
    compile(ast.Module(body=[func_node], type_ignores=[]), filename=str(train_path), mode="exec"),
    module.__dict__,
)

discrete_mapping = module.discrete_mapping


def test_discrete_mapping_symmetry_and_range():
    np.random.seed(0)
    matrix = np.array([[0, 0.2, 0.3], [0.2, 0, 0.5], [0.3, 0.5, 0]])
    mapped = discrete_mapping(matrix)

    # Assert symmetry
    assert np.allclose(mapped, mapped.T)

    # Assert all values within [0, 1]
    assert np.min(mapped) >= 0 and np.max(mapped) <= 1


def test_discrete_mapping_reproducibility():
    matrix = np.array([[0, 0.2, 0.3], [0.2, 0, 0.5], [0.3, 0.5, 0]])

    np.random.seed(0)
    mapped1 = discrete_mapping(matrix)

    np.random.seed(0)
    mapped2 = discrete_mapping(matrix)

    assert np.array_equal(mapped1, mapped2)
