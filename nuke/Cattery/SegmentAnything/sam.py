from PySide2 import QtWidgets, QtOpenGL

import nuke

SAM_TABLE_TOOLTIP = "<b>tracks</b><br>sam"


def find_widget_by_tooltip(tooltip):
    """Retrieve a QWidget by its text Tooltip"""
    stack = QtWidgets.QApplication.instance().allWidgets()
    while stack:
        widget = stack.pop()
        if widget.toolTip() == tooltip:
            return widget


def find_widget_by_text(text):
    """Retrieve a QLineEdit by its text value"""
    text = str(text)
    for widget in QtWidgets.QApplication.topLevelWidgets():
        for line_edit in widget.findChildren(QtWidgets.QLineEdit):
            if line_edit.text() == text:
                return line_edit.window()


def get_widget_from_node(node_name: str) -> QtOpenGL.QGLWidget:
    """Retrieve the QGLWidget of DAG graph"""
    stack = QtWidgets.QApplication.instance().allWidgets()
    while stack:
        widget = stack.pop()
        if widget.objectName() == node_name:
            return widget


def hide_columns():
    table_widget = find_widget_by_tooltip(SAM_TABLE_TOOLTIP)
    table_view = table_widget.findChild(QtWidgets.QTableView)

    # Hide columns 'offset', 'T, 'R' and 'error'
    for i in (4, 5, 6, 7, 9):
        table_view.setColumnHidden(i, True)


def sam():
    if nuke.thisKnob().name() == "showPanel":
        node_name = nuke.thisNode().name()

        # Open the tracker node panel
        nuke.toNode(f"{node_name}.Tracker1").showControlPanel(True)

        # Hide the tracker node widget
        unique_id = nuke.thisNode().knob("unique_id").value()
        unique_id = int(unique_id + 1)
        tracker_window = find_widget_by_text(unique_id)
        if tracker_window:
            tracker_window.hide()

        table_widget = find_widget_by_tooltip(SAM_TABLE_TOOLTIP)
        table_view = table_widget.findChild(QtWidgets.QTableView)
        model = table_view.model()

        # Hide the unneeded columns; make sure it stays hidden on any refresh.
        hide_columns()
        model.modelAboutToBeReset.connect(lambda: hide_columns())
        model.modelReset.connect(lambda: hide_columns())
        model.dataChanged.connect(lambda: hide_columns())

    if nuke.thisKnob().name() == "hidePanel":
        node_name = nuke.thisNode().name()

        table_widget = find_widget_by_tooltip(SAM_TABLE_TOOLTIP)
        table_view = table_widget.findChild(QtWidgets.QTableView)
        model = table_view.model()

        try:
            model.modelAboutToBeReset.disconnect()
            model.modelReset.disconnect()
            model.dataChanged.disconnect()
        except RuntimeError:
            pass

        # Close the tracker node panel
        nuke.toNode(f"{node_name}.Tracker1").hideControlPanel()
