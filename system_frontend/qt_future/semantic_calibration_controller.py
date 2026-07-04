"""Controller for the semantic calibration Designer UI.

UI layout lives in semantic_calibration_page.ui.
This file only loads the UI and connects behavior.
"""

from __future__ import annotations

import sys
import os
import re
from pathlib import Path

import PyQt5
from PyQt5 import uic
from PyQt5.QtCore import QEvent, QLibraryInfo, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QTextCharFormat, QTextOption
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QPushButton, QStyle, QTextEdit, QVBoxLayout, QWidget


UI_PATH = Path(__file__).with_name("semantic_calibration_page.ui")
DOMAIN_DESC_DIR = Path(__file__).with_name("domain_desc")
CLASS_DESC_DIR = Path(__file__).with_name("class_desc")
PLACEHOLDER_TEXT = "-- Selection --"
ALL_DOMAINS = [
    "GPR-SD",
    "GPR-Road",
    "A1 Sandy Loam",
    "A2 Saturated Silty Clay",
    "A3 Urban Backfill",
    "A4 Layered Road",
]
DOMAIN_NAME_ALIASES = {
    "GPR-SD": ["GPR-SD", "GPR_SD", "GPR SD", "SD"],
    "GPR-Road": ["GPR-Road", "GPR_Road", "GPR Road", "Road"],
    "A1 Sandy Loam": ["A1 Sandy Loam", "A1"],
    "A2 Saturated Silty Clay": ["A2 Saturated Silty Clay", "A2"],
    "A3 Urban Backfill": ["A3 Urban Backfill", "A3"],
    "A4 Layered Road": ["A4 Layered Road", "A4"],
}


def configure_qt_platform_plugin() -> None:
    """Use the Qt platform plugins from the active Python environment.

    This avoids hardcoding Anaconda paths and lets PyCharm virtualenvs run the
    UI as long as PyQt5 is installed in that environment.
    """
    pyqt_root = Path(PyQt5.__file__).resolve().parent
    qt5_root = pyqt_root / "Qt5"
    qt_bin = qt5_root / "bin"
    qt_plugins = qt5_root / "plugins"
    platforms = qt_plugins / "platforms"

    if qt_bin.exists():
        os.environ["PATH"] = f"{qt_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(qt_bin))

    if not platforms.exists():
        plugin_root = Path(QLibraryInfo.location(QLibraryInfo.PluginsPath))
        platforms = plugin_root / "platforms"

    if platforms.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
        os.environ["QT_PLUGIN_PATH"] = str(platforms.parent)


DOMAIN_DESCRIPTIONS = {
    "GPR-SD": "GPR subsurface diagnosis data collected from concrete and asphalt roads. The domain is relatively structured and contains B-scan evidence for cavity, crack, loose, normal, and pipeline classes.",
    "GPR-Road": "GPR B-scan images collected from concrete, asphalt, and unpaved road surfaces. The domain contains stronger surface diversity, clutter, and road-condition variation.",
    "A1 Sandy Loam": "A dry, low-loss simulated area with high sand content and limited interference. Radar reflections are usually clearer and less attenuated.",
    "A2 Saturated Silty Clay": "A high-loss simulated area with strong water content and clay-dominant soil. Radar energy attenuates more quickly and deeper evidence becomes blurred.",
    "A3 Urban Backfill": "A heterogeneous simulated backfill area with random scatterers. B-scan signatures can be noisy and locally fragmented.",
    "A4 Layered Road": "A layered road-structure area with asphalt and gravel layers. Layer interfaces introduce structured interference and additional reflection bands.",
}


CLASS_HINTS = {
    "cavity": "air-filled voids, strong dielectric contrast, local diffraction, and abrupt reflection changes",
    "crack": "thin discontinuities, local breaks in layered reflections, and narrow scattering responses",
    "loose": "poorly compacted material, diffuse scattering, attenuation, and unstable texture",
    "normal": "continuous stratigraphy, stable background texture, and absence of localized abnormal reflections",
    "pipeline": "regular buried object geometry, hyperbolic reflections, and strong material contrast",
}

REAL_DOMAIN_CLASSES = ["Cavity", "Crack", "Loose", "Normal", "Pipeline"]
SIMULATED_DOMAIN_CLASSES = ["Cavity", "Crack", "Pipeline"]
SIMULATED_DOMAINS = {"A1 Sandy Loam", "A2 Saturated Silty Clay", "A3 Urban Backfill", "A4 Layered Road"}
REAL_DOMAINS = {"GPR-SD", "GPR-Road"}
MAX_UNDO_STEPS = 20
MIN_HYPHEN_PREFIX = 5
MIN_HYPHEN_SUFFIX = 4
MIN_FILL_SPLIT_WORD_LENGTH = 10

class SemanticCalibrationController(QMainWindow):
    """Backend/controller layer for semantic_calibration_page.ui."""

    def __init__(self):
        super().__init__()
        uic.loadUi(str(UI_PATH), self)
        self.active_editor = self.sourceClassDescriptionTextEdit
        self._restoring_state = False
        self._domain_options_updating = False
        self._undo_stack = []
        self._last_state = None
        self._active_paint_color = None
        self._saved_expert_correction = None
        self._set_app_icon()
        self._apply_style()
        self._set_initial_empty_state()
        self._fix_layout_stretch()
        self._create_io_buttons()
        self._create_loading_overlay()
        self._bind_events()
        self._last_state = self._capture_state()

    def _set_app_icon(self) -> None:
        icon = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self.setWindowIcon(icon)
        self.topIconLabel.setPixmap(icon.pixmap(18, 18))

    def _fix_layout_stretch(self) -> None:
        self.classComboBox.setFixedWidth(180)
        self.generateDescriptionButton.setMinimumSize(200, 34)
        self.imageAssistedGenerationCheckBox.setFixedWidth(230)
        self.classLabel.setFixedHeight(30)
        self.sourceDomainDescriptionTextEdit.setFixedHeight(72)
        self.targetDomainDescriptionTextEdit.setFixedHeight(72)
        self.sourceClassDescriptionTextEdit.setMinimumHeight(130)
        self.targetClassDescriptionTextEdit.setMinimumHeight(130)
        self.expertRevisionTextEdit.setFixedHeight(32)
        self.correctButton.setFixedHeight(34)
        self.wrongButton.setFixedHeight(34)
        self.addButton.setFixedHeight(34)
        self._disable_text_scrollbars()
        self.generationControlLayout.setStretch(0, 0)
        self.generationControlLayout.setStretch(1, 0)
        self.generationControlLayout.setStretch(2, 1)
        self.generationControlLayout.setStretch(3, 0)
        self.generationControlLayout.setStretch(4, 0)
        self.generationControlLayout.setStretch(5, 0)
        self._normalize_button_fonts()

    def _normalize_button_fonts(self) -> None:
        normal_button_font = QFont("Arial", 10)
        normal_button_font.setStyleStrategy(QFont.PreferAntialias)
        normal_button_font.setStretch(QFont.Unstretched)

        generate_font = QFont("Arial", 10)
        generate_font.setBold(True)
        generate_font.setStyleStrategy(QFont.PreferAntialias)
        generate_font.setStretch(QFont.Unstretched)
        self.generateDescriptionButton.setFont(generate_font)

        for button in [
            self.correctButton,
            self.wrongButton,
            self.addButton,
            self.undoButton,
            self.resetButton,
        ]:
            button.setFont(normal_button_font)

    def _set_initial_empty_state(self) -> None:
        self._set_text_placeholders()
        for combo in [
            self.sourceDomainComboBox,
            self.targetDomainComboBox,
        ]:
            self._set_combo_items(combo, ALL_DOMAINS, "")

        self.classComboBox.clear()
        self.classComboBox.addItem(PLACEHOLDER_TEXT)
        self.classComboBox.setCurrentIndex(0)
        self.imageAssistedGenerationCheckBox.setChecked(False)

        for combo in [
            self.sourceDomainComboBox,
            self.targetDomainComboBox,
            self.classComboBox,
        ]:
            combo.setCurrentIndex(0)

        for editor in [
            self.sourceDomainDescriptionTextEdit,
            self.targetDomainDescriptionTextEdit,
            self.sourceClassDescriptionTextEdit,
            self.targetClassDescriptionTextEdit,
            self.expertRevisionTextEdit,
        ]:
            editor.clear()

    def _apply_style(self) -> None:
        QApplication.instance().setFont(QFont("Arial", 10))
        self.setStyleSheet(
            """
            QWidget { font-family: Arial; color: #1f2933; background: #f4f6f8; }
            QLabel#titleLabel { color: #17202a; padding: 8px; }
            QLabel#classLabel { background: white; font-weight: 600; font-size: 15px; padding: 0 0; }
            QWidget#navPanel { background: #202a36; border-radius: 10px; }
            QListWidget#navigationList { background: transparent; color: #d7dee8; border: 0; padding: 0; outline: 0; }
            QListWidget#navigationList::item { height: 36px; padding-left: 10px; border-radius: 6px; }
            QListWidget#navigationList::item:hover { background: #2d3948; color: white; }
            QListWidget#navigationList::item:selected { background: #3b82f6; color: white; border: 0; outline: 0; }
            QListWidget#navigationList::item:selected:hover { background: #4f8df8; }
            QListWidget#navigationList::item:pressed { background: #2563eb; padding-left: 12px; padding-top: 1px; }
            QListWidget#navigationList::item:focus { border: 0; outline: 0; }
            QGroupBox { background: white; border: 1px solid #cfd8e3; border-radius: 8px; margin-top: 10px; padding: 14px 8px 8px 8px; font-size: 15px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; top: 0px; padding: 0 5px; background: #f4f6f8; font-size: 15px; font-weight: 600; }
            QTextEdit, QComboBox { background: #fbfcfd; border: 1px solid #cfd6df; border-radius: 6px; padding: 6px; font-size: 13px; font-weight: 400; }
            QTextEdit:focus, QComboBox:focus { border: 1px solid #2f8f83; background: white; }
            QTextEdit#expertRevisionTextEdit { padding: 5px 8px; font-size: 13px; }
            QTextEdit QScrollBar:vertical { width: 0px; }
            QTextEdit QScrollBar:horizontal { height: 0px; }
            QFrame#generationControlFrame { background: white; border: 1px solid #d8dee6; border-radius: 8px; }
            QPushButton { background: #4b5563; color: white; border: 0; border-radius: 6px; padding: 5px 12px; font-size: 13px; font-weight: 600; outline: 0; }
            QPushButton:hover { background: #5b6675; }
            QPushButton:pressed { background: #374151; padding-left: 13px; padding-top: 7px; padding-right: 11px; padding-bottom: 5px; }
            QPushButton#generateDescriptionButton { background: #0f766e; font-size: 13px; font-weight: 600; padding: 5px 12px; }
            QPushButton#generateDescriptionButton:hover { background: #12867d; }
            QPushButton#generateDescriptionButton:pressed { background: #0d5f59; padding-left: 13px; padding-top: 7px; padding-right: 11px; padding-bottom: 5px; }
            QCheckBox#imageAssistedGenerationCheckBox { background: white; padding: 4px 2px; font-size: 13px; font-weight: 500; spacing: 7px; }
            QPushButton#correctButton { background: #1f9d55; }
            QPushButton#correctButton:hover { background: #25b463; }
            QPushButton#correctButton:pressed { background: #177d42; padding-left: 13px; padding-top: 7px; padding-right: 11px; padding-bottom: 5px; }
            QPushButton#correctButton:checked { background: #166534; border: 2px solid #bbf7d0; padding: 4px 10px; }
            QPushButton#wrongButton { background: #d64545; }
            QPushButton#wrongButton:hover { background: #e05252; }
            QPushButton#wrongButton:pressed { background: #b93636; padding-left: 13px; padding-top: 7px; padding-right: 11px; padding-bottom: 5px; }
            QPushButton#wrongButton:checked { background: #991b1b; border: 2px solid #fecaca; padding: 4px 10px; }
            QPushButton#undoButton { background: #334155; }
            QPushButton#undoButton:hover { background: #42526a; }
            QPushButton#undoButton:pressed { background: #253247; padding-left: 13px; padding-top: 7px; padding-right: 11px; padding-bottom: 5px; }
            QPushButton#resetButton { background: #7f1d1d; }
            QPushButton#resetButton:hover { background: #991f1f; }
            QPushButton#resetButton:pressed { background: #651515; padding-left: 13px; padding-top: 7px; padding-right: 11px; padding-bottom: 5px; }
            QPushButton#ioToolButton {
                background: #f8fafc;
                color: #1f2933;
                border: 1px solid #c8d3df;
                border-radius: 5px;
                padding: 3px 8px;
                font-size: 13px;
                font-weight: 400;
            }
            QPushButton#ioToolButton:hover { background: #eef6ff; border: 1px solid #60a5fa; }
            QPushButton#ioToolButton:pressed { background: #dbeafe; padding-left: 8px; padding-top: 4px; padding-right: 6px; padding-bottom: 2px; }
            QWidget#loadingOverlay { background: rgba(17, 24, 39, 150); }
            QLabel#loadingOverlayLabel {
                background: white;
                color: #111827;
                border: 1px solid #d8dee6;
                border-radius: 8px;
                padding: 18px 42px;
                font-size: 18px;
                font-weight: 700;
            }
            """
        )

    def _create_loading_overlay(self) -> None:
        self.loadingOverlay = QWidget(self.centralwidget)
        self.loadingOverlay.setObjectName("loadingOverlay")
        self.loadingOverlay.hide()

        layout = QVBoxLayout(self.loadingOverlay)
        layout.setAlignment(Qt.AlignCenter)
        self.loadingOverlayLabel = QLabel("Generating...")
        self.loadingOverlayLabel.setObjectName("loadingOverlayLabel")
        self.loadingOverlayLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loadingOverlayLabel)
        self._sync_loading_overlay_geometry()

    def _create_io_buttons(self) -> None:
        self.sourceDomainUploadButton = self._make_io_button(self.sourceDomainGroupBox, "Upload", QStyle.SP_ArrowUp)
        self.targetDomainUploadButton = self._make_io_button(self.targetDomainGroupBox, "Upload", QStyle.SP_ArrowUp)
        self.sourceClassDownloadButton = self._make_io_button(
            self.sourceClassDescriptionGroupBox,
            "Download",
            QStyle.SP_ArrowDown,
        )
        self.targetClassDownloadButton = self._make_io_button(
            self.targetClassDescriptionGroupBox,
            "Download",
            QStyle.SP_ArrowDown,
        )
        self._io_button_groups = [
            (self.sourceDomainGroupBox, self.sourceDomainDescriptionTextEdit, self.sourceDomainUploadButton),
            (self.targetDomainGroupBox, self.targetDomainDescriptionTextEdit, self.targetDomainUploadButton),
            (self.sourceClassDescriptionGroupBox, self.sourceClassDescriptionTextEdit, self.sourceClassDownloadButton),
            (self.targetClassDescriptionGroupBox, self.targetClassDescriptionTextEdit, self.targetClassDownloadButton),
        ]
        self.expertSaveButton = self._make_io_button(
            self.expertCorrectionGroupBox,
            "Save",
            QStyle.SP_DialogSaveButton,
            width=92,
        )
        for target, anchor_editor, _button in self._io_button_groups:
            target.installEventFilter(self)
            anchor_editor.installEventFilter(self)
        self.expertCorrectionGroupBox.installEventFilter(self)
        self.expertRevisionTextEdit.installEventFilter(self)
        self._position_io_buttons()
        QTimer.singleShot(0, self._position_io_buttons)

    def _make_io_button(self, parent: QWidget, text: str, icon_id, width: int = 116) -> QPushButton:
        button = QPushButton(text, parent)
        button.setObjectName("ioToolButton")
        font = QFont("Arial", 10)
        font.setStretch(QFont.Unstretched)
        font.setStyleStrategy(QFont.PreferAntialias)
        button.setFont(font)
        button.setIcon(self.style().standardIcon(icon_id))
        button.setIconSize(QSize(13, 13))
        button.setFixedSize(width, 28)
        button.setCursor(Qt.PointingHandCursor)
        button.setToolTip(text)
        button.raise_()
        return button

    def _position_io_buttons(self) -> None:
        if not hasattr(self, "_io_button_groups"):
            return
        title_y = 1
        for target, anchor_editor, button in self._io_button_groups:
            right_edge = anchor_editor.geometry().right()
            if right_edge <= button.width():
                right_edge = target.width() - 10
            button.move(max(right_edge - button.width() + 1, 0), title_y)
            button.raise_()
        if hasattr(self, "expertSaveButton"):
            right_edge = self.expertRevisionTextEdit.geometry().right()
            if right_edge <= self.expertSaveButton.width():
                right_edge = self.expertCorrectionGroupBox.width() - 10
            self.expertSaveButton.move(max(right_edge - self.expertSaveButton.width() + 1, 0), title_y)
            self.expertSaveButton.raise_()

    def _sync_loading_overlay_geometry(self) -> None:
        if hasattr(self, "loadingOverlay"):
            self.loadingOverlay.setGeometry(self.centralwidget.rect())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._sync_loading_overlay_geometry()
        self._position_io_buttons()
        QTimer.singleShot(0, self._position_io_buttons)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._position_io_buttons)

    def _bind_events(self) -> None:
        self.navigationList.setCurrentRow(0)
        self.navigationList.setFocusPolicy(Qt.NoFocus)
        self.correctButton.setCheckable(True)
        self.wrongButton.setCheckable(True)
        self.sourceDomainComboBox.currentTextChanged.connect(self.handle_domain_change)
        self.targetDomainComboBox.currentTextChanged.connect(self.handle_domain_change)
        self.classComboBox.currentTextChanged.connect(self.handle_class_change)
        self.imageAssistedGenerationCheckBox.stateChanged.connect(self.handle_option_change)
        self.generateDescriptionButton.clicked.connect(self.start_description_generation)
        self.sourceDomainUploadButton.clicked.connect(
            lambda: self.upload_domain_text(self.sourceDomainComboBox, self.sourceDomainDescriptionTextEdit)
        )
        self.targetDomainUploadButton.clicked.connect(
            lambda: self.upload_domain_text(self.targetDomainComboBox, self.targetDomainDescriptionTextEdit)
        )
        self.sourceClassDownloadButton.clicked.connect(
            lambda: self.download_class_description(
                self.sourceDomainComboBox,
                self.classComboBox,
                self.sourceClassDescriptionTextEdit,
            )
        )
        self.targetClassDownloadButton.clicked.connect(
            lambda: self.download_class_description(
                self.targetDomainComboBox,
                self.classComboBox,
                self.targetClassDescriptionTextEdit,
            )
        )
        self.correctButton.clicked.connect(lambda checked: self.set_paint_tool(QColor("#c8f7dc"), "correct", checked))
        self.wrongButton.clicked.connect(lambda checked: self.set_paint_tool(QColor("#ffd1d1"), "wrong", checked))
        self.addButton.clicked.connect(self.add_expert_revision)
        self.expertSaveButton.clicked.connect(self.save_expert_correction)
        self.undoButton.clicked.connect(self.undo_last_operation)
        self.resetButton.clicked.connect(self.reset_right_page)
        self.sourceClassDescriptionTextEdit.selectionChanged.connect(
            lambda: self.set_active_editor(self.sourceClassDescriptionTextEdit)
        )
        self.targetClassDescriptionTextEdit.selectionChanged.connect(
            lambda: self.set_active_editor(self.targetClassDescriptionTextEdit)
        )
        self.sourceClassDescriptionTextEdit.viewport().installEventFilter(self)
        self.targetClassDescriptionTextEdit.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Resize:
            QTimer.singleShot(0, self._position_io_buttons)
        if event.type() == QEvent.MouseButtonRelease:
            for editor in [self.sourceClassDescriptionTextEdit, self.targetClassDescriptionTextEdit]:
                if obj is editor.viewport():
                    self.apply_paint_tool_to_selection(editor)
                    break
        return super().eventFilter(obj, event)

    def _disable_text_scrollbars(self) -> None:
        for editor in [
            self.sourceDomainDescriptionTextEdit,
            self.targetDomainDescriptionTextEdit,
            self.sourceClassDescriptionTextEdit,
            self.targetClassDescriptionTextEdit,
            self.expertRevisionTextEdit,
        ]:
            editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            editor.setViewportMargins(0, 0, 0, 0)
            editor.document().setDocumentMargin(2)
            editor.verticalScrollBar().setFixedWidth(0)
            editor.horizontalScrollBar().setFixedHeight(0)
            editor.verticalScrollBar().hide()
            editor.horizontalScrollBar().hide()
            editor.setLineWrapMode(QTextEdit.WidgetWidth)
            self._format_text_editor(editor)

    def _format_text_editor(self, editor: QTextEdit) -> None:
        option = QTextOption()
        option.setAlignment(Qt.AlignLeft)
        option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        editor.document().setDefaultTextOption(option)
        editor.setAlignment(Qt.AlignLeft)

    def _set_text_placeholders(self) -> None:
        self.sourceDomainDescriptionTextEdit.setPlaceholderText("Editable description of the selected original area.")
        self.targetDomainDescriptionTextEdit.setPlaceholderText("Editable description of the selected new area.")
        self.sourceClassDescriptionTextEdit.setPlaceholderText("Generated description for the selected class in the original area.")
        self.targetClassDescriptionTextEdit.setPlaceholderText("Generated description for the selected class in the new area.")
        self.expertRevisionTextEdit.setPlaceholderText("Type expert revision or additional evidence here.")

    def _set_wrapped_text(self, editor: QTextEdit, text: str) -> None:
        editor.setPlainText(self._wrap_text_to_editor(editor, text))
        editor.setAlignment(Qt.AlignLeft)
        self._scroll_editor_to_top(editor)

    def _scroll_editor_to_top(self, editor: QTextEdit) -> None:
        cursor = editor.textCursor()
        cursor.movePosition(cursor.Start)
        editor.setTextCursor(cursor)
        editor.verticalScrollBar().setValue(0)
        editor.horizontalScrollBar().setValue(0)

    def _wrap_text_to_editor(self, editor: QTextEdit, text: str) -> str:
        max_width = self._available_text_width(editor)
        return "\n".join(
            self._wrap_paragraph_by_pixels(editor, paragraph, max_width)
            for paragraph in text.splitlines()
        )

    def _available_text_width(self, editor: QTextEdit) -> int:
        viewport_width = editor.viewport().width()
        widget_width = editor.width()
        raw_width = max(viewport_width, widget_width - 8, 120)
        return max(raw_width - 18, 120)

    def _wrap_paragraph_by_pixels(self, editor: QTextEdit, paragraph: str, max_width: int) -> str:
        words = re.findall(r"\S+", paragraph)
        if not words:
            return ""

        lines = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if self._text_width(editor, candidate) <= max_width:
                current = candidate
                continue

            if current:
                split = self._split_word_to_fill_line(editor, current, word, max_width)
                if split:
                    filled_line, word = split
                    lines.append(filled_line)
                else:
                    lines.append(current)
                current = ""

            if not word:
                continue

            if self._text_width(editor, word) <= max_width:
                current = word
            else:
                pieces = self._split_word_by_pixels(editor, word, max_width)
                lines.extend(pieces[:-1])
                current = pieces[-1]

        if current:
            lines.append(current)
        return "\n".join(lines)

    def _split_word_to_fill_line(
        self,
        editor: QTextEdit,
        current: str,
        word: str,
        max_width: int,
    ) -> tuple[str, str] | None:
        if len(word.strip(".,;:()[]{}")) < MIN_FILL_SPLIT_WORD_LENGTH:
            return None

        best_prefix = ""
        for index in range(MIN_HYPHEN_PREFIX, len(word) - MIN_HYPHEN_SUFFIX + 1):
            prefix = f"{word[:index]}-"
            candidate = f"{current} {prefix}"
            if self._text_width(editor, candidate) <= max_width:
                best_prefix = prefix
            else:
                break

        if not best_prefix:
            return None

        suffix_start = len(best_prefix) - 1
        return f"{current} {best_prefix}", word[suffix_start:]

    def _split_word_by_pixels(self, editor: QTextEdit, word: str, max_width: int) -> list[str]:
        pieces = []
        current = ""
        for index, char in enumerate(word):
            candidate = f"{current}{char}"
            remaining = len(word) - index
            can_split = remaining >= MIN_HYPHEN_SUFFIX
            if current and can_split and self._text_width(editor, f"{candidate}-") > max_width:
                pieces.append(f"{current}-")
                current = char
            else:
                current = candidate
        if current:
            pieces.append(current)
        return pieces

    def _text_width(self, editor: QTextEdit, text: str) -> int:
        return editor.fontMetrics().horizontalAdvance(text)

    def set_active_editor(self, editor: QTextEdit) -> None:
        self.active_editor = editor

    def _set_combo_items(self, combo, items: list[str], selected: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(PLACEHOLDER_TEXT)
        combo.addItems(items)
        if selected and selected in items:
            combo.setCurrentText(selected)
        else:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _domain_family(self, domain: str) -> str:
        if domain in REAL_DOMAINS:
            return "real"
        if domain in SIMULATED_DOMAINS:
            return "simulated"
        return ""

    def _domain_options_for_peer(self, peer_domain: str) -> list[str]:
        family = self._domain_family(peer_domain)
        if family == "real":
            return [domain for domain in ALL_DOMAINS if domain in REAL_DOMAINS]
        if family == "simulated":
            return [domain for domain in ALL_DOMAINS if domain in SIMULATED_DOMAINS]
        return ALL_DOMAINS

    def _refresh_domain_options(self) -> None:
        if self._domain_options_updating:
            return
        self._domain_options_updating = True
        source_domain = self._selected_text(self.sourceDomainComboBox)
        target_domain = self._selected_text(self.targetDomainComboBox)
        source_options = self._domain_options_for_peer(target_domain or source_domain)
        target_options = self._domain_options_for_peer(source_domain or target_domain)
        self._set_combo_items(
            self.sourceDomainComboBox,
            source_options,
            source_domain,
        )
        self._set_combo_items(
            self.targetDomainComboBox,
            target_options,
            target_domain,
        )
        self._domain_options_updating = False

    def _capture_state(self) -> dict:
        return {
            "source_domain": self._selected_text(self.sourceDomainComboBox),
            "target_domain": self._selected_text(self.targetDomainComboBox),
            "class_name": self._selected_text(self.classComboBox),
            "image_assisted": self.imageAssistedGenerationCheckBox.isChecked(),
            "source_domain_description": self.sourceDomainDescriptionTextEdit.toPlainText(),
            "target_domain_description": self.targetDomainDescriptionTextEdit.toPlainText(),
            "source_class_description": self.sourceClassDescriptionTextEdit.toHtml(),
            "target_class_description": self.targetClassDescriptionTextEdit.toHtml(),
            "expert_revision": self.expertRevisionTextEdit.toPlainText(),
        }

    def _push_undo_state(self) -> None:
        state = self._capture_state()
        snapshot = self._last_state or state
        if not self._undo_stack or self._undo_stack[-1] != snapshot:
            self._undo_stack.append(snapshot)
            self._undo_stack = self._undo_stack[-MAX_UNDO_STEPS:]

    def _sync_last_state(self) -> None:
        self._last_state = self._capture_state()

    def _restore_state(self, state: dict) -> None:
        self._restoring_state = True
        self._set_combo_items(self.sourceDomainComboBox, ALL_DOMAINS, state["source_domain"])
        self._set_combo_items(self.targetDomainComboBox, ALL_DOMAINS, state["target_domain"])
        self._refresh_domain_options()
        self.update_class_options()
        self._set_combo_items(
            self.classComboBox,
            [self.classComboBox.itemText(i) for i in range(1, self.classComboBox.count())],
            state["class_name"],
        )
        self.imageAssistedGenerationCheckBox.setChecked(state["image_assisted"])
        self._set_wrapped_text(self.sourceDomainDescriptionTextEdit, state["source_domain_description"])
        self._set_wrapped_text(self.targetDomainDescriptionTextEdit, state["target_domain_description"])
        self.sourceClassDescriptionTextEdit.setHtml(state["source_class_description"])
        self.targetClassDescriptionTextEdit.setHtml(state["target_class_description"])
        self.expertRevisionTextEdit.setPlainText(state["expert_revision"])
        self._scroll_editor_to_top(self.sourceClassDescriptionTextEdit)
        self._scroll_editor_to_top(self.targetClassDescriptionTextEdit)
        self._restoring_state = False
        self._sync_last_state()

    def handle_domain_change(self) -> None:
        if self._restoring_state or self._domain_options_updating:
            return
        self._push_undo_state()
        self._refresh_domain_options()
        self.update_class_options()
        self._sync_last_state()

    def handle_class_change(self) -> None:
        if self._restoring_state:
            return
        self._push_undo_state()
        self.clear_class_descriptions()
        self._sync_last_state()

    def handle_option_change(self) -> None:
        if self._restoring_state:
            return
        self._push_undo_state()
        self._sync_last_state()

    def upload_domain_text(self, combo, editor: QTextEdit) -> None:
        DOMAIN_DESC_DIR.mkdir(exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Domain Description",
            str(DOMAIN_DESC_DIR),
            "Text Files (*.txt);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog,
        )
        if not file_path:
            return

        try:
            selected_path = Path(file_path)
            raw_text = self._read_text_file(selected_path)
        except OSError as exc:
            QMessageBox.warning(self, "Upload Failed", f"Unable to read the selected file.\n{exc}")
            return

        domain, description = self._parse_domain_text(raw_text)
        if domain not in ALL_DOMAINS:
            domain = self._detect_domain_name(selected_path.stem)
        if not description:
            QMessageBox.warning(self, "Empty File", "The selected text file does not contain usable description content.")
            return

        self._push_undo_state()
        if domain in ALL_DOMAINS:
            combo.blockSignals(True)
            combo.setCurrentText(domain)
            combo.blockSignals(False)
            self._refresh_domain_options()
            self.update_class_options()
        self._set_wrapped_text(editor, description)
        self.clear_class_descriptions()
        self._sync_last_state()
        self.statusbar.showMessage(f"Loaded domain description from {selected_path.name}.", 3000)

    def _read_text_file(self, path: Path) -> str:
        data = path.read_bytes()
        for encoding in ["utf-8-sig", "utf-8", "gbk", "cp936"]:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    def download_class_description(self, domain_combo, class_combo, editor: QTextEdit) -> None:
        description = editor.toPlainText().strip()
        if not description:
            QMessageBox.warning(self, "Empty Description", "There is no class description to download.")
            return

        default_name = self._default_class_description_filename(
            self._selected_text(domain_combo),
            self._selected_text(class_combo),
        )
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Download Class Description",
            str(Path(__file__).resolve().parent / default_name),
            "Text Files (*.txt);;All Files (*)",
            options=QFileDialog.DontUseNativeDialog,
        )
        if not file_path:
            return

        save_path = Path(file_path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".txt")

        try:
            save_path.write_text(description, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Download Failed", f"Unable to save the description.\n{exc}")
            return

        self.statusbar.showMessage(f"Saved class description to {save_path.name}.", 3000)

    def _default_class_description_filename(self, domain: str, class_name: str) -> str:
        if domain and class_name:
            return f"{domain}_{class_name}.txt"
        if domain:
            return f"{domain}_ClassDescription.txt"
        if class_name:
            return f"{class_name}_ClassDescription.txt"
        return "ClassDescription.txt"

    def _parse_domain_text(self, text: str) -> tuple[str, str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return "", ""

        domain = ""
        description_lines = []
        in_description = False

        for line in normalized.splitlines():
            stripped = line.strip()
            if not stripped:
                if in_description:
                    description_lines.append("")
                continue

            field_match = re.match(r"^(domain|area|name)\s*:\s*(.+)$", stripped, re.IGNORECASE)
            if field_match:
                candidate = field_match.group(2).strip()
                detected_domain = self._detect_domain_name(candidate)
                if detected_domain in ALL_DOMAINS:
                    domain = detected_domain
                    continue

            description_match = re.match(r"^(description|domain description|area description)\s*:\s*(.*)$", stripped, re.IGNORECASE)
            if description_match:
                in_description = True
                first_value = description_match.group(2).strip()
                if first_value:
                    description_lines.append(first_value)
                continue

            if in_description:
                description_lines.append(stripped)
            else:
                description_lines.append(stripped)

        description = "\n".join(description_lines).strip()
        return domain, description

    def _detect_domain_name(self, raw_name: str) -> str:
        plain_name = raw_name.lower()
        compact_name = self._compact_domain_token(raw_name)
        if not compact_name:
            return ""

        candidates = []
        for domain, aliases in DOMAIN_NAME_ALIASES.items():
            for alias in aliases:
                compact_alias = self._compact_domain_token(alias)
                if not compact_alias:
                    continue
                if len(compact_alias) <= 2:
                    pattern = rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])"
                    if not re.search(pattern, plain_name):
                        continue
                elif compact_alias not in compact_name:
                    continue
                candidates.append((len(compact_alias), domain))
                break

        if not candidates:
            return ""
        candidates.sort(reverse=True)
        return candidates[0][1]

    def _compact_domain_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def undo_last_operation(self) -> None:
        if not self._undo_stack:
            return
        self._restore_state(self._undo_stack.pop())

    def reset_right_page(self) -> None:
        self._push_undo_state()
        self._restoring_state = True
        self._set_initial_empty_state()
        self._restoring_state = False
        self._sync_last_state()

    def clear_class_descriptions(self) -> None:
        self.sourceClassDescriptionTextEdit.clear()
        self.targetClassDescriptionTextEdit.clear()

    def update_domain_descriptions(self) -> None:
        self.clear_class_descriptions()

    def update_class_options(self) -> None:
        current = self._selected_text(self.classComboBox)
        source_classes = self._classes_for_domain(self._selected_text(self.sourceDomainComboBox))
        target_classes = self._classes_for_domain(self._selected_text(self.targetDomainComboBox))

        if source_classes and target_classes:
            allowed = [cls for cls in source_classes if cls in target_classes]
        else:
            allowed = source_classes or target_classes or []

        self.classComboBox.blockSignals(True)
        self.classComboBox.clear()
        self.classComboBox.addItem(PLACEHOLDER_TEXT)
        self.classComboBox.addItems(allowed)
        if current and current.title() in allowed:
            self.classComboBox.setCurrentText(current.title())
        else:
            self.classComboBox.setCurrentIndex(0)
            self.sourceClassDescriptionTextEdit.clear()
            self.targetClassDescriptionTextEdit.clear()
        self.classComboBox.blockSignals(False)

    def _classes_for_domain(self, domain: str) -> list[str]:
        if not domain:
            return []
        if domain in SIMULATED_DOMAINS:
            return SIMULATED_DOMAIN_CLASSES
        if domain in REAL_DOMAINS:
            return REAL_DOMAIN_CLASSES
        return REAL_DOMAIN_CLASSES

    def start_description_generation(self) -> None:
        current_class = self._selected_text(self.classComboBox)
        source_domain = self._selected_text(self.sourceDomainComboBox)
        target_domain = self._selected_text(self.targetDomainComboBox)
        missing_fields = []
        if not source_domain:
            missing_fields.append("original area")
        if not target_domain:
            missing_fields.append("new area")
        if not current_class:
            missing_fields.append("class")
        if missing_fields:
            self.sourceClassDescriptionTextEdit.clear()
            self.targetClassDescriptionTextEdit.clear()
            QMessageBox.warning(
                self,
                "Selection Required",
                f"Select {', '.join(missing_fields)} before generating descriptions.",
            )
            return

        self._push_undo_state()
        self.generateDescriptionButton.setEnabled(False)
        self.loadingOverlay.show()
        self.loadingOverlay.raise_()
        QTimer.singleShot(1800, self.generate_descriptions)

    def generate_descriptions(self) -> None:
        current_class = self._selected_text(self.classComboBox)
        source_domain = self._selected_text(self.sourceDomainComboBox)
        target_domain = self._selected_text(self.targetDomainComboBox)
        if not current_class or not source_domain or not target_domain:
            self.sourceClassDescriptionTextEdit.clear()
            self.targetClassDescriptionTextEdit.clear()
            self.loadingOverlay.hide()
            self.generateDescriptionButton.setEnabled(True)
            return

        source_text, source_error = self._load_class_description_file(source_domain, current_class)
        target_text, target_error = self._load_class_description_file(target_domain, current_class)

        self.loadingOverlay.hide()
        self.generateDescriptionButton.setEnabled(True)
        if source_error or target_error:
            QMessageBox.warning(
                self,
                "Description File Missing",
                "\n".join(error for error in [source_error, target_error] if error),
            )
            return

        self._set_wrapped_text(self.sourceClassDescriptionTextEdit, source_text)
        self._set_wrapped_text(self.targetClassDescriptionTextEdit, target_text)
        self._sync_last_state()

    def _load_class_description_file(self, domain: str, class_name: str) -> tuple[str, str]:
        file_path = CLASS_DESC_DIR / f"{domain}_{class_name}.txt"
        if not file_path.exists():
            return "", f"Missing file: {file_path.name}"

        try:
            description = self._read_text_file(file_path).strip()
        except OSError as exc:
            return "", f"Unable to read {file_path.name}: {exc}"

        if not description:
            return "", f"Empty file: {file_path.name}"
        return description, ""

    def _image_assisted_suffix(self) -> str:
        image_hint = ""
        if self.imageAssistedGenerationCheckBox.isChecked():
            image_hint = " Image-assisted evidence is enabled and can be used as an auxiliary cue when available."
        return image_hint

    def _build_source_class_description(self, class_name: str, domain: str) -> str:
        domain_effect = DOMAIN_DESCRIPTIONS.get(domain, "The domain condition is not specified, so the description uses generic GPR evidence.")
        return (
            f"Original {class_name.title()} Evidence in {domain}\n\n"
            f"Source evidence is used as the semantic reference. "
            f"Main cues include {CLASS_HINTS[class_name]}. "
            f"Domain context: {domain_effect} "
            f"The wording should anchor the classifier to repeatable source-domain patterns."
            f"{self._image_assisted_suffix()}"
        )

    def _build_target_class_description(self, class_name: str, domain: str, source_domain: str) -> str:
        domain_effect = DOMAIN_DESCRIPTIONS.get(domain, "The domain condition is not specified, so the description uses generic GPR evidence.")
        transfer_context = (
            f"Compared with {source_domain}, "
            if source_domain and source_domain != "Unspecified Source Domain"
            else "For the deployment area, "
        )
        return (
            f"New-Area {class_name.title()} Evidence in {domain}\n\n"
            f"{transfer_context}target evidence is recalibrated for the new area. "
            f"Expected cues still involve {CLASS_HINTS[class_name]}, but strength, texture, and continuity may shift. "
            f"Domain context: {domain_effect} "
            f"The wording should support target-domain recognition."
            f"{self._image_assisted_suffix()}"
        )

    def _selected_text(self, combo) -> str:
        text = combo.currentText().strip()
        return "" if text == PLACEHOLDER_TEXT else text

    def set_paint_tool(self, color: QColor, mode: str, checked: bool) -> None:
        if not checked:
            self.clear_paint_tool()
            return

        self._active_paint_color = color
        self.correctButton.setChecked(mode == "correct")
        self.wrongButton.setChecked(mode == "wrong")
        self.sourceClassDescriptionTextEdit.viewport().setCursor(Qt.CrossCursor)
        self.targetClassDescriptionTextEdit.viewport().setCursor(Qt.CrossCursor)

    def clear_paint_tool(self) -> None:
        self._active_paint_color = None
        self.correctButton.setChecked(False)
        self.wrongButton.setChecked(False)
        self.sourceClassDescriptionTextEdit.viewport().unsetCursor()
        self.targetClassDescriptionTextEdit.viewport().unsetCursor()

    def apply_paint_tool_to_selection(self, editor: QTextEdit) -> None:
        if self._active_paint_color is None:
            return
        cursor = editor.textCursor()
        if not cursor.hasSelection():
            return
        self.apply_highlight(editor, self._active_paint_color)

    def mark_selection(self, color: QColor) -> None:
        editor = self.active_editor
        if not isinstance(editor, QTextEdit):
            return
        self.apply_highlight(editor, color)

    def apply_highlight(self, editor: QTextEdit, color: QColor) -> None:
        cursor = editor.textCursor()
        if not cursor.hasSelection():
            QMessageBox.warning(self, "No Text Selected", "Select text in a class description first.")
            return
        self._push_undo_state()
        fmt = QTextCharFormat()
        fmt.setBackground(color)
        cursor.mergeCharFormat(fmt)
        cursor.clearSelection()
        editor.setTextCursor(cursor)
        self._sync_last_state()

    def add_expert_revision(self) -> None:
        text = self.expertRevisionTextEdit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Empty Revision", "Enter expert revision text first.")
            return
        self._push_undo_state()
        cursor = self.active_editor.textCursor()
        if cursor.hasSelection():
            cursor.insertText(text)
        else:
            cursor.movePosition(cursor.End)
            cursor.insertText(f"\n\nExpert revision: {text}")
        self.expertRevisionTextEdit.clear()
        self._sync_last_state()

    def save_expert_correction(self) -> None:
        self._saved_expert_correction = {
            "source_domain": self._selected_text(self.sourceDomainComboBox),
            "target_domain": self._selected_text(self.targetDomainComboBox),
            "class_name": self._selected_text(self.classComboBox),
            "source_class_description_html": self.sourceClassDescriptionTextEdit.toHtml(),
            "target_class_description_html": self.targetClassDescriptionTextEdit.toHtml(),
            "pending_expert_revision": self.expertRevisionTextEdit.toPlainText().strip(),
            "image_assisted": self.imageAssistedGenerationCheckBox.isChecked(),
        }
        self.statusbar.showMessage("Expert correction saved.", 3000)


def main() -> int:
    configure_qt_platform_plugin()
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 10))
    window = SemanticCalibrationController()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
