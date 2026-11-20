"""Space-group equivalent reflections using the simulation lattice backend."""
import ast
import math

from PyQt5.QtWidgets import (
    QDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)

from ...sample import LatticeStructure


def parse_reflections(text):
    value = ast.literal_eval(text)
    if isinstance(value, (list, tuple)) and len(value) == 3 and all(type(v) is int for v in value):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Enter an integer HKL triplet or a list of triplets.")
    reflections = []
    for hkl in value:
        if (not isinstance(hkl, (list, tuple)) or len(hkl) != 3
                or any(type(v) is not int for v in hkl) or not any(hkl)):
            raise ValueError("Each reflection must have three integer indices, excluding (0, 0, 0).")
        if tuple(hkl) not in reflections:
            reflections.append(tuple(hkl))
    return reflections


class EquivalentReflectionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Equivalent Reflections")
        self.resize(800, 640)
        layout = QVBoxLayout(self)
        self.use_sample_button = QPushButton("Use Current Sample")
        layout.addWidget(self.use_sample_button)
        form = QGridLayout()
        self.fields = {}
        for name, label, default in (
            ("space_group", "Space group:", "167"),
            ("a", "a [\u00c5]:", "4.954"),
            ("b", "b [\u00c5]:", "4.954"),
            ("c", "c [\u00c5]:", "14.01"),
            ("alpha", "\u03b1 [\u00b0]:", "90"),
            ("beta", "\u03b2 [\u00b0]:", "90"),
            ("gamma", "\u03b3 [\u00b0]:", "120"),
            ("hkl", "HKL(s):", "(1, 1, 0)"),
        ):
            field = QLineEdit(default)
            self.fields[name] = field
            if name in ("space_group", "hkl"):
                row = 0 if name == "space_group" else 3
                form.addWidget(QLabel(label), row, 0)
                form.addWidget(field, row, 1, 1, 5)
            else:
                names = ("a", "b", "c") if name in ("a", "b", "c") else ("alpha", "beta", "gamma")
                row = 1 if name in ("a", "b", "c") else 2
                col = 2 * names.index(name)
                form.addWidget(QLabel(label), row, col)
                form.addWidget(field, row, col + 1)
        self.fields["hkl"].setPlaceholderText("[(1, 0, 0), (2, 0, 0)]")
        layout.addLayout(form)
        actions = QHBoxLayout()
        self.calculate_button = QPushButton("Find Equivalent Reflections")
        self.calculate_button.clicked.connect(self.calculate)
        actions.addWidget(self.calculate_button)
        layout.addLayout(actions)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(240)
        layout.addWidget(self.output, stretch=1)

    def values(self):
        return {name: field.text() for name, field in self.fields.items()}

    def set_values(self, values):
        for name, value in values.items():
            if name in self.fields:
                self.fields[name].setText(str(value))
        self.output.clear()
        self.status.clear()

    def calculate(self):
        try:
            values = self.values()
            inputs = parse_reflections(values.pop("hkl"))
            space_group = int(values.pop("space_group"))
            if not 1 <= space_group <= 230:
                raise ValueError("Space group must be between 1 and 230.")
            cell = {name: float(value) for name, value in values.items()}
            if any(not math.isfinite(v) or v <= 0 for v in cell.values()):
                raise ValueError("Cell parameters must be positive finite numbers.")
            if any(cell[name] >= 180 for name in ("alpha", "beta", "gamma")):
                raise ValueError("Cell angles must be smaller than 180 degrees.")
            lattice = LatticeStructure(space_group=space_group, **cell)
            results = {
                hkl: sorted({tuple(int(v) for v in row)
                             for row in lattice.get_equivalent_reflections(hkl)})
                for hkl in inputs
            }
            self.output.setPlainText("\n\n".join(
                f"{hkl}: {len(reflections)} equivalents\n{reflections}"
                for hkl, reflections in results.items()))
            unique = set().union(*(set(rows) for rows in results.values()))
            self.status.setText(f"{len(results)} input reflection(s); {len(unique)} unique equivalent(s)")
            return results[inputs[0]] if len(inputs) == 1 else results
        except Exception as exc:
            self.output.clear()
            self.status.setText(f"Error: {exc}")
            return None
