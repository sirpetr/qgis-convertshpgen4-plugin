# -*- coding: utf-8 -*-
"""
/***************************************************************************
 ShapefileToGen4
                                 A QGIS plugin
 This plugin convert Shapefiles to Gen4 for John Deere.
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2023-12-29
        git sha              : $Format:%H$
        copyright       ,     : (C) 2023 by Petr Sirucek
        email                : psirucek@seznam.cz
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""
import json
import time
import io
import os
import threading

from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication
from qgis.PyQt.QtGui import QIcon, QFont, QTextCharFormat
from qgis.PyQt.QtWidgets import QAction, QFileDialog

# Initialize Qt resources from file resources.py
from .resources import *
# Import the code for the dialog
from .shapefile_gen4_dialog import ShapefileToGen4Dialog
import os.path
from pathlib import Path
from qgis.core import QgsProject, Qgis, QgsMapLayer, QgsWkbTypes, QgsApplication, QgsTask, QgsMessageLog
import subprocess
import logging
import datetime
from PyQt5.QtCore import QTimer
from .creating_Gen4 import create_file

from qgis.core import (
    QgsApplication, QgsTask, QgsMessageLog, Qgis
    )

MESSAGE_CATEGORY = 'TaskFromFunction'


def now_time(john_deere=True):
    """At this time in string"""

    if john_deere:
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        now = datetime.datetime.now()
        date_time = now.strftime("%H:%M:%S")

    return date_time


class ShapefileToGen4:
    """QGIS Plugin Implementation."""

    def __init__(self, iface):
        """Constructor.

        :param iface: An interface instance that will be passed to this class
            which provides the hook by which you can manipulate the QGIS
            application at run time.
        :type iface: QgsInterface
        """
        # Save reference to the QGIS interface
        self.iface = iface
        # initialize plugin directory
        self.plugin_dir = os.path.dirname(__file__)
        # initialize locale
        locale = QSettings().value('locale/userLocale')[0:2]
        locale_path = os.path.join(
            self.plugin_dir,
            'i18n',
            'ShapefileToGen4_{}.qm'.format(locale))

        if os.path.exists(locale_path):
            self.translator = QTranslator()
            self.translator.load(locale_path)
            QCoreApplication.installTranslator(self.translator)

        # Declare instance attributes
        self.actions = []
        self.menu = self.tr(u'&John Deere Tools')

        # Check if plugin was started the first time in current QGIS session
        # Must be set in initGui() to survive plugin reloads
        self.layer_polygon_select = None
        self.layer_line_select = None

        # Create the dialog with elements (after translation) and keep reference
        self.dlg = None
        self.timer = QTimer()
        self.tasks = []

        # Folder of plugin
        self.path_home = os.path.realpath(__file__)
        self.path_home = os.path.dirname(self.path_home)
        self.path_home = self.path_home.replace('\\', '/')

    def tr(self, message):
        """Get the translation for a string using Qt translation API.

        We implement this ourselves since we do not inherit QObject.

        :param message: String for translation.
        :type message: str, QString

        :returns: Translated version of message.
        :rtype: QString
        """
        # noinspection PyTypeChecker,PyArgumentList,PyCallByClass
        return QCoreApplication.translate('ShapefileToGen4', message)

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        whats_this=None,
        parent=None):
        """Add a toolbar icon to the toolbar.

        :param icon_path: Path to the icon for this action. Can be a resource
            path (e.g. ':/plugins/foo/bar.png') or a normal file system path.
        :type icon_path: str

        :param text: Text that should be shown in menu items for this action.
        :type text: str

        :param callback: Function to be called when the action is triggered.
        :type callback: function

        :param enabled_flag: A flag indicating if the action should be enabled
            by default. Defaults to True.
        :type enabled_flag: bool

        :param add_to_menu: Flag indicating whether the action should also
            be added to the menu. Defaults to True.
        :type add_to_menu: bool

        :param add_to_toolbar: Flag indicating whether the action should also
            be added to the toolbar. Defaults to True.
        :type add_to_toolbar: bool

        :param status_tip: Optional text to show in a popup when mouse pointer
            hovers over the action.
        :type status_tip: str

        :param parent: Parent widget for the new action. Defaults None.
        :type parent: QWidget

        :param whats_this: Optional text to show in the status bar when the
            mouse pointer hovers over the action.

        :returns: The action that was created. Note that the action is also
            added to self.actions list.
        :rtype: QAction
        """

        icon = QIcon(icon_path)
        action = QAction(icon, text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if whats_this is not None:
            action.setWhatsThis(whats_this)

        if add_to_toolbar:
            # Adds plugin icon to Plugins toolbar
            self.iface.addToolBarIcon(action)

        if add_to_menu:
            self.iface.addPluginToVectorMenu(
                self.menu,
                action)

        self.actions.append(action)

        return action

    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI."""

        icon_path = ':/plugins/shapefile_gen4/icon.png'
        self.add_action(
            icon_path,
            text=self.tr(u'Convert Shapefiles to Gen4'),
            callback=self.run,
            parent=self.iface.mainWindow())

    def unload(self):
        """Removes the plugin menu item and icon from QGIS GUI."""
        for action in self.actions:
            self.iface.removePluginVectorMenu(
                self.tr(u'&John Deere Tools'),
                action)
            self.iface.removeToolBarIcon(action)

    def on_combobox_1_changed(self, index):
        """This method is called when the selection in the comboBox changes"""

        selected_value = self.dlg.comboBox_1.itemText(index)
        # Change all another Combox

        if len(selected_value):
            self.layer_polygon_select = QgsProject.instance().mapLayersByName(selected_value)[0]
            list_fields = [None] + [field.name() for field in self.layer_polygon_select.fields() if field.typeName() == 'String']

            self.dlg.comboBox_3.clear()
            self.dlg.comboBox_3.addItems(list_fields)
            self.dlg.comboBox_4.clear()
            self.dlg.comboBox_4.addItems(list_fields)
            self.dlg.comboBox_5.clear()
            self.dlg.comboBox_5.addItems(list_fields)
            self.dlg.comboBox_6.clear()
            self.dlg.comboBox_6.addItems(list_fields)

    def on_combobox_2_changed(self, index):
        """This method is called when the selection in the comboBox changes"""

        # Change all another Combox
        selected_value_line = self.dlg.comboBox_2.itemText(index)
        self.dlg.comboBox_7.clear()
        self.dlg.comboBox_8.clear()

        # List column lines
        if len(selected_value_line):
            self.layer_line_select = QgsProject.instance().mapLayersByName(selected_value_line)[0]

            # For line names
            list_fields = [None] + [field.name() for field in self.layer_line_select.fields() if field.typeName() == 'String']
            self.dlg.comboBox_7.addItems(list_fields)

            # For line types
            int_types = ['Integer', 'Integer64', 'Float', 'Float64']
            list_type_fields = [None] + [field.name() for field in self.layer_line_select.fields() if field.typeName() in int_types]
            self.dlg.comboBox_8.addItems(list_type_fields)
        else:
            self.layer_line_select = None

    def on_button_clicked(self):

        self.dlg.tabWidget.setCurrentIndex(1)
        self.dlg.progressBar.reset()
        if not self.dlg.comboBox_1.currentText() or \
            not self.dlg.comboBox_3.currentText() or \
            not self.dlg.comboBox_4.currentText() or \
            not self.dlg.comboBox_5.currentText():
            self.dlg.textEdit_2.append(f'<a>{now_time(john_deere=False)} - Missing input values!</a>')
        elif self.dlg.comboBox_3.currentText() == self.dlg.comboBox_4.currentText():
            self.dlg.textEdit_2.append(f'<a>{now_time(john_deere=False)} - Do not input the same name of client and farm!</a>')
        else:
            self.dlg.textEdit_2.clear()
            self.dlg.progressBar.setFormat("Running...")
            os.chdir(self.path_home)

            if self.layer_line_select is not None:
                path_line_selected = self.layer_line_select.source()
            else:
                path_line_selected = ''

            # Input data
            input_json = {
                'path_home': self.path_home,
                'json_metadat': 'gen4_scripts/base_element/version1/metadata.json',
                'shpFields': self.layer_polygon_select.source().replace('\\', '/'),
                'shpGuide': path_line_selected.replace('\\', '/'),
                'clientName': self.dlg.comboBox_3.currentText(),  # 'CLIENT_NAM',
                'farmColumn': self.dlg.comboBox_4.currentText(),  # 'FARM_NAME',
                'fieldColumn': self.dlg.comboBox_5.currentText(),  # FIELD_NAME',
                'cropColumn': self.dlg.comboBox_6.currentText(),  # 'CROP',
                'lineColumn': self.dlg.comboBox_7.currentText(),
                'lineColumnType': self.dlg.comboBox_8.currentText(),
            }

            # Save input Json
            with open('gen4_scripts/input.json', 'w', encoding='utf-8') as j:
                json.dump(input_json, j, indent=4)
            globals()['task1'] = QgsTask.fromFunction('Convert Shapefiles to Gen4',
                                                      convert_gen4_task,
                                                      input_json=input_json,
                                                      text_edit=self.dlg.textEdit_2,
                                                      progressBar=self.dlg.progressBar,
                                                      path_home=self.path_home)
            task = globals()['task1']
            QgsApplication.taskManager().addTask(task)
            self.tasks.append(task)

    def func_end(self):

        # some code for your algorithm
        self.dlg.close()
        self.dlg = None

    def clear_text(self):
        # Clear the text in the QTextEdit widget
        self.dlg.textEdit_2.clear()

    def run(self):
        """Run method that performs all the real work"""

        current_path = os.path.realpath(__file__)
        logging.basicConfig(filename=os.path.join(os.path.dirname(current_path), 'log.log'), level='INFO')
        self.dlg = ShapefileToGen4Dialog()

        # Signal and slot settings
        self.dlg.textEdit.setReadOnly(True)
        self.dlg.textEdit_2.setReadOnly(True)
        self.dlg.comboBox_1.currentIndexChanged.connect(self.on_combobox_1_changed)
        self.dlg.comboBox_2.currentIndexChanged.connect(self.on_combobox_2_changed)
        self.dlg.progressBar.reset()

        self.dlg.comboBox_3.setEditable(True)
        self.dlg.comboBox_4.setEditable(True)

        # Populate the comboBox with names of all the loaded layers
        layers = QgsProject.instance().mapLayers().values()
        layers_vector = [layer for layer in layers if layer.type() == QgsMapLayer.VectorLayer]

        # comboBox_1 - Polygon layers
        layers_polygon = [vector.name() for vector in layers_vector if vector.geometryType() == QgsWkbTypes.PolygonGeometry]
        self.dlg.comboBox_1.clear()
        self.dlg.comboBox_1.addItems(layers_polygon)

        # comboBox_2 - Line layers
        layers_line = [vector.name() for vector in layers_vector if vector.geometryType() == QgsWkbTypes.LineGeometry]
        self.dlg.comboBox_2.clear()
        self.dlg.comboBox_2.addItems([""] + layers_line)

        # Run process
        self.clear_text()
        self.dlg.pushButton_3.clicked.connect(self.on_button_clicked)

        # Closed application
        self.dlg.pushButton_4.clicked.connect(self.func_end)

        # show the dialog
        self.dlg.show()

        # Run the dialog event loop
        result = self.dlg.exec_()

        # See if OK was pressed
        if result:
            pass


def convert_gen4_task(task, input_json, text_edit, progressBar, path_home):
    """
    Raises an exception to abort the task.
    Returns a result if success.
    The result will be passed, together with the exception (None in
    the case of success), to the on_finished method.
    If there is an exception, there will be no result.
    """

    QgsMessageLog.logMessage('Started task {}'.format(task.description()), MESSAGE_CATEGORY, Qgis.Info)
    text_edit.append(f'<a>{now_time(john_deere=False)} - Running...</a>')

    progressBar.setRange(0, 0)

    try:
        # Subprocess
        output_info, output_folder = create_file(input_json, None)  # None
        output_info = [f"<p>{x}</p>" for x in output_info]
        output_info = ''.join(output_info)
        output_info = """<style>p {margin: 0; padding: 0;}</style>""" + output_info
        text_edit.append(output_info)

        # Control output process in HTML
        file_path = os.path.join(path_home, "log_output.html")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(output_info)

    except Exception as e:
        print("An error occurred:", e)
        text_edit.append(f'<a>{now_time(john_deere=False)} - {e}</a>')
        logging.info("An error occurred:", e)

    # Check if there were any error
    output_folder = output_folder.replace(r'\\', '/')
    if output_folder:
        output_folder = os.path.dirname(output_folder)
        text_edit.append(f"""<a>{now_time(john_deere=False)} - Output folder:</a>""")
        text_edit.append(f"""<a style="color: blue; text-decoration: underline;">{output_folder}</a>""")

    text_edit.append(f'<a>{now_time(john_deere=False)} - Gen4 completed!</a>')
    progressBar.setRange(0, 100)

    return {'task': task.description()}


def stopped(task):
    QgsMessageLog.logMessage(
        'Task "{name}" was canceled'.format(
            name=task.description()),
        MESSAGE_CATEGORY, Qgis.Info)


def completed(exception, result=None):
    """This is called when doSomething is finished.
    Exception is not None if doSomething raises an exception.
    result is the return value of doSomething."""
    if exception is None:
        if result is None:
            QgsMessageLog.logMessage(
                'Completed with no exception and no result ' \
                '(probably manually canceled by the user)',
                MESSAGE_CATEGORY, Qgis.Warning)
        else:
            QgsMessageLog.logMessage(
                'Task {name} completed'.format(
                    name=result['task']), MESSAGE_CATEGORY, Qgis.Info)
    else:
        QgsMessageLog.logMessage("Exception: {}".format(exception),
                                 MESSAGE_CATEGORY, Qgis.Critical)
        raise exception







