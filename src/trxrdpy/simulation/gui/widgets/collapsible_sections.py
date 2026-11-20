"""Collapsible form sections that preserve enabled states and field values."""
from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QGroupBox, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QObject):
    changed = pyqtSignal()

    def __init__(self, group, key):
        super().__init__(group)
        self.group = group
        self.key = key
        self.collapsed = False
        self.expanded_minimum_height = group.minimumHeight()
        self.content = QWidget(group)
        self.content.setObjectName("sectionContents")
        self.content.setStyleSheet("QWidget#sectionContents { background: transparent; }")
        # Reparent the existing layout, preserving nested controls' visibility.
        self.content.setLayout(group.layout())
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.content)
        self.button = QToolButton(group)
        self.button.setFixedSize(24, 24)
        self.button.setStyleSheet("QToolButton { background: transparent; border: none; }")
        self.button.clicked.connect(lambda: self.set_collapsed(not self.collapsed))
        group.installEventFilter(self)
        self.set_collapsed(False)

    def eventFilter(self, watched, event):
        if watched is self.group and event.type() == QEvent.Resize:
            self.button.move(max(0, self.group.width() - 34), 2)
            self.button.raise_()
        return super().eventFilter(watched, event)

    def set_collapsed(self, collapsed):
        changed = self.collapsed != bool(collapsed)
        self.collapsed = bool(collapsed)
        self.group.setProperty("sectionCollapsed", self.collapsed)
        self.group.style().unpolish(self.group)
        self.group.style().polish(self.group)
        self.group.setMinimumHeight(
            max(self.expanded_minimum_height, self.group.fontMetrics().height() + 18)
            if self.collapsed else self.expanded_minimum_height)
        self.content.setVisible(not self.collapsed)
        self.button.setArrowType(Qt.RightArrow if self.collapsed else Qt.DownArrow)
        action = "Expand" if self.collapsed else "Collapse"
        self.button.setToolTip(f"{action} {self.group.title()}")
        self.button.setAccessibleName(self.button.toolTip())
        self.group.updateGeometry()
        if changed:
            self.changed.emit()


def install_collapsible_sections(root, prefix):
    sections = []
    for group in root.findChildren(QGroupBox):
        if not group.title() or group.layout() is None:
            continue
        titles = [group.title()]
        parent = group.parentWidget()
        while parent is not None and parent is not root:
            if isinstance(parent, QGroupBox) and parent.title():
                titles.insert(0, parent.title())
            parent = parent.parentWidget()
        sections.append(CollapsibleSection(group, prefix + "/" + "/".join(titles)))
    return sections
