import math
import os
import sys

import matplotlib
import pyvista

sys.path.append("src")

from subt_proc_gen.param_classes import TunnelPtClGenParams
from EasyConfig import EasyConfig
from subt_proc_gen.display_functions import (
    plot_graph,
    plot_splines,
    plot_mesh,
    plot_tunnel_ptcls,
    plot_intersection_ptcls,
)
from subt_proc_gen.mesh_generation import (
    TunnelNetworkMeshGenerator,
    TunnelNetworkPtClGenParams,
    TunnelNetworkMeshGenParams,
)

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtGui import QPixmap, QPainter, QPen, QIcon
from PyQt5.QtWidgets import (
    QMenuBar,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QFrame,
    QTabWidget,
    QSplitter,
    QLabel,
    QFileDialog,
    QToolBar,
    QInputDialog,
    QMenu,
    QSlider,
    QScrollArea,
    QProgressDialog,
)
from pyvista import QtInteractor
from pyvistaqt import QtInteractor
from subt_proc_gen.tunnel import *
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QRect, QPoint, QRunnable, QThreadPool, QPointF
import logging as log

log.basicConfig(level=log.DEBUG)


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class Point(QPoint):
    def __init__(self, pose=None, index=0):
        super().__init__()
        if pose is not None:
            self.setX(pose.x())
            self.setY(pose.y())
            self.index = index
            self.zpos = 0

    def z(self):
        return self.zpos

    def setZ(self, val):
        self.zpos = val

    def nppose(self):
        return np.array([self.x(), self.y(), self.z()])

    def serialize(self, dictionary):
        dictionary["index"] = self.index
        dictionary["x"] = self.x()
        dictionary["y"] = self.y()
        dictionary["z"] = self.z()

    def deserialize(self, dictionary):
        self.index = dictionary["index"]
        self.setX(dictionary["x"])
        self.setY(dictionary["y"])
        self.setZ(dictionary["z"])


class Sketch(QLabel):
    def __init__(self):
        super().__init__()
        pix = QPixmap(200, 200)
        pix.fill(Qt.white)
        self.color_index = 0
        self.colors = [
            Qt.red,
            Qt.green,
            Qt.blue,
            Qt.darkYellow,
            Qt.darkMagenta,
            Qt.black,
            Qt.yellow,
            Qt.gray,
            Qt.cyan,
        ]
        self.setPixmap(pix)
        self.setScaledContents(True)
        self.base = None
        self.p1 = None
        self.p2 = None
        self.points = []
        self.lines = []
        self.trees = []
        self.scale = 0.1
        self.current_tree_index = None
        self.selected = None
        self.params = {}

    def load(self, filename):
        self.points.clear()
        self.current_tree_index = 0
        self.trees.clear()
        try:
            f = open(filename, "r")
            data = yaml.safe_load(f)
            f.close()

            if len(data["points"]) > 0:
                for p in data["points"]:
                    q = Point()
                    q.deserialize(p)
                    self.points.append(q)
            print("Loaded {} points".format(len(self.points)))

            for color, tree in enumerate(data["trees"]):
                tree_nodes = []
                for node_id in tree["nodes"]:
                    tree_nodes.append(self.points[node_id])
                self.trees.append({"nodes": tree_nodes, "color": color, "config": tree["config"]})
            self.color_index = len(self.trees) - 1
            self.update()
            return True
        except:
            return False

    def save(self, filename):
        filename = "default.yaml" if filename is None else filename

        dictionary = {"points": [], "trees": [], "config": []}

        for p in self.points:
            data = {}
            p.serialize(data)
            dictionary["points"].append(data)

        for tree in self.trees:
            dictionary["trees"].append(
                {
                    "nodes": [node.index for node in tree.get("nodes")],
                    "config": tree.get("config"),
                }
            )

        with open(filename, "w") as f:
            yaml.dump(dictionary, f)

    def update_defaults(self, **kwargs):
        self.params = kwargs
        self.update()

    def current_tree(self):
        return self.trees[self.current_tree_index]

    def delete_last(self):
        nodes = self.current_tree().get("nodes")
        if len(nodes) > 1:
            point = nodes.pop(-1)
            found = False
            for tree in self.trees:
                found = found or (point in tree.get("nodes"))
            if not found:
                self.points.remove(point)
                for i, node in enumerate(self.points):
                    node.index = i

        if len(nodes) > 0:
            self.p1 = nodes[-1]
        else:
            self.current_tree_index = 0
            self.p1 = None
            self.p2 = None

        self.update()

    def setScale(self, scale):
        self.scale = scale

    def clear_points(self):
        self.points.clear()
        self.trees.clear()
        self.current_tree_index = None
        self.p1 = None
        self.p2 = None
        self.color_index = 0
        self.update()

    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent) -> None:
        for p in self.points:
            r = QRect(p.x() - 10, p.y() - 10, 20, 20)
            if r.contains(ev.pos().x(), ev.pos().y()):
                selected_tree = None
                for tree in self.trees:
                    nodes, tunnel_config = tree.get("nodes"), tree.get("config")
                    if p in nodes:
                        selected_tree = tree
                        break

                qm = QMenu()

                set_z, cont = qm.addAction("Set Z"), None
                change_config, cont = qm.addAction("Change Tunnel Config"), None
                if p == selected_tree.get("nodes")[-1]:
                    cont = qm.addAction("Continue here")

                res = qm.exec(ev.globalPos())

                if res == cont:
                    self.p1 = p
                    self.current_tree_index = self.trees.index(selected_tree)
                    self.color_index = selected_tree.get("color")
                    return

                elif res == set_z:
                    text, ok = QInputDialog.getText(
                        self,
                        "Set Z coordinate of node " + str(p.index),
                        "Value",
                        text=str(p.z()),
                    )

                    if ok and text.replace(".", "").replace("-", "").isnumeric():
                        p.setZ(float(text))
                        self.update()

                elif res == change_config:
                    dialog = self.Config()
                    dialog.set("roughness", selected_tree.get("config").get("roughness", 0))
                    dialog.set("radius", selected_tree.get("config").get("radius", 0))
                    if dialog.edit():
                        selected_tree.get("config")["roughness"] = dialog.get("roughness")
                        selected_tree.get("config")["radius"] = dialog.get("radius")

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseDoubleClickEvent(ev)
        for p in self.points:
            r = QRect(p.x() - 10, p.y() - 10, 20, 20)
            if r.contains(ev.pos().x(), ev.pos().y()):
                text, ok = QInputDialog.getText(
                    self,
                    "Set Z coordinate of node " + str(p.index),
                    "Value",
                    text=str(p.z()),
                )
                if ok and text.replace(".", "").replace("-", "").isnumeric():
                    p.setZ(float(text))
                    self.update()
                break

    class Config(EasyConfig):
        def __init__(self):
            super().__init__()
            self.root().addSlider("roughness", pretty="Roughness", val=0, min=0, max=10, den=10)
            self.root().addSlider("radius", pretty="Radius", val=3, min=10, max=100, den=10)

    def getPoints(self, scale):
        self.p1 = None
        self.p2 = None
        self.update()

        if len(self.points) == 0:
            return

        nodes = []
        for p in self.points:
            nodes.append(Node(p.nppose() * scale))

        x, y, z = nodes[0].x, nodes[0].y, nodes[0].z
        for n in nodes:
            n.set_pose(-x + n.x, y - n.y, -z + n.z)

        trees_list = list()

        for tree in self.trees:
            tree_nodes = list()
            for p in tree.get("nodes"):
                tree_nodes.append(nodes[p.index])
            trees_list.append({"tree_nodes": tree_nodes, "tree_config": tree.get("config")})
        return trees_list
        # return nodes, node_trees

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mousePressEvent(ev)
        if ev.modifiers() & Qt.ControlModifier:
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    self.selected = p

        elif ev.modifiers() & Qt.ShiftModifier:
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    self.color_index += 1
                    self.trees.append(
                        {
                            "nodes": [p],
                            "color": self.color_index,
                            "config": self.params.copy(),
                        }
                    )
                    self.current_tree_index = len(self.trees) - 1
                    self.p1 = p

        elif self.p1 is None and len(self.trees) == 0:
            self.base = ev.pos()
            self.p1 = Point(ev.pos(), 0)
            self.trees.append(
                {
                    "nodes": [self.p1],
                    "color": self.color_index,
                    "config": self.params.copy(),
                }
            )
            self.points.append(self.p1)
            self.current_tree_index = 0

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseMoveEvent(ev)
        if self.selected:
            self.selected.setX(ev.pos().x())
            self.selected.setY(ev.pos().y())
            self.update()
        elif self.p1 is not None:
            self.p2 = ev.pos()
            self.update()

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        super().mouseReleaseEvent(ev)
        if self.selected:
            self.selected = None
        elif self.p1 and self.p2:
            found = None
            for p in self.points:
                r = QRect(p.x() - 10, p.y() - 10, 20, 20)
                if r.contains(ev.pos().x(), ev.pos().y()):
                    p.setX(self.p2.x())
                    p.setY(self.p2.y())
                    found = p
                    break
            if found:
                self.current_tree().get("nodes").append(found)
                self.p1 = found
            else:
                self.p1 = Point(self.p2, len(self.points))
                self.current_tree().get("nodes").append(self.p1)
                self.points.append(self.p1)
            self.p2 = None
            self.update()

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)

        for tree in self.trees:
            nodes, color = tree.get("nodes"), tree.get("color")
            for i, p in enumerate(nodes):
                painter.setPen(Qt.black)
                if p.z() != 0:
                    painter.drawText(
                        QPoint(p.x() + 10, p.y() + 10),
                        str(p.index) + " z:" + str(p.z()),
                    )
                else:
                    painter.drawText(QPoint(p.x() + 10, p.y() + 10), str(p.index))

                painter.setPen(self.colors[color])
                painter.drawEllipse(p, 10, 10)
                if i > 0:
                    painter.drawLine(p, nodes[i - 1])
                    p1, p2 = p, nodes[i - 1]
                    dist = math.sqrt(math.pow(p1.x() - p2.x(), 2) + math.pow(p1.y() - p2.y(), 2))
                    painter.setPen(Qt.black)
                    painter.drawText(
                        QPointF((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2),
                        "{:.1f}".format(float(dist) * self.params.get("scale", 1)),
                    )

        if self.p1 and self.p2:
            painter.setPen(QPen(self.colors[self.color_index]))
            painter.drawLine(self.p1, self.p2)
            dist = math.sqrt(math.pow(self.p1.x() - self.p2.x(), 2) + math.pow(self.p1.y() - self.p2.y(), 2))
            painter.setPen(Qt.black)
            painter.drawText(self.p2, "{:.1f}".format(float(dist) * self.params.get("scale", 1)))

        if self.p1:
            painter.setPen(QPen(self.colors[self.color_index]))
            painter.drawEllipse(self.p1, 15, 15)

        if len(self.points) > 0:
            painter.setPen(QPen(Qt.red))
            painter.drawLine(
                self.rect().x(),
                self.points[0].y(),
                self.rect().width() + self.rect().x(),
                self.points[0].y(),
            )
            painter.setPen(QPen(Qt.green))
            painter.drawLine(
                self.points[0].x(),
                self.rect().y(),
                self.points[0].x(),
                self.rect().height() + self.rect().y(),
            )

    def get_trees(self):
        return self.trees


class MainWindow(QtWidgets.QMainWindow):
    model_config = '<?xml version="1.0"?>\n\
<model>\n\
  <name>$name$</name>\n\
  <version>1.0</version>\n \
  <sdf version="1.6">model.sdf</sdf>\n\
  <description>\n\
  </description>\n\
</model>'

    model_sdf = '<?xml version="1.0"?>\n\
<sdf version="1.6">\n\
  <model name="$name$">\n\
    <static>true</static>\n\
    <link name="link">\n\
      <pose>0 0 0 0 0 0</pose>\n\
      <collision name="collision">\n\
        <geometry>\n\
          <mesh>\n\
            <uri>./mesh.obj</uri>\n\
          </mesh>\n\
        </geometry>\n\
      </collision>\n\
      <visual name="visual">\n\
        <geometry>\n\
          <mesh>\n\
            <uri>./mesh.obj</uri>\n\
          </mesh>\n\
        </geometry>\n\
      </visual>\n\
    </link>\n\
  </model>\n\
</sdf>'

    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        self.lay.setStretchFactor(0, 1)
        self.lay.setStretchFactor(1, 10)
        # self.sc.figure.set_figwidth(self.geometry().width())
        # self.sc.figure.set_figheight(self.geometry().height())

    def update_defaults(self):
        self.sketch.update_defaults(
            radius=self.radius_slider.value(),
            roughness=self.slider.value() / 10,
            floor=self.floor_slider.value(),
            scale=self.scale_slider.value() / 10,
        )

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.pd = None
        self.mesh_generator = None
        self.config = {}
        menu = QMenuBar()
        self.setMenuBar(menu)

        m1 = menu.addMenu("File")
        m1.addAction("Load", self.load_yaml)
        m1.addAction("New", self.new_yaml)
        m1.addAction("Save as ", self.save_yaml)

        m2 = menu.addMenu("Edit")
        m2.addAction("Config", self.edit)

        self.lay = QSplitter()
        self.lay.setStretchFactor(0, 1)
        self.lay.setStretchFactor(1, 10)

        self.sketch = Sketch()

        cb = QPushButton("Clear")
        cb.clicked.connect(self.sketch.clear_points)

        self.frame2 = QFrame()
        self.plotter2 = QtInteractor(self.frame2)

        vlayout = QVBoxLayout()

        render_tb = QToolBar()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setTickInterval(100)

        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setValue(4)
        self.radius_slider.setMinimum(1)
        self.radius_slider.setMaximum(8)
        self.radius_slider.setTickInterval(1)

        self.floor_slider = QSlider(Qt.Horizontal)
        self.floor_slider.setMinimum(-4)
        self.floor_slider.setMaximum(0)
        self.floor_slider.setTickInterval(1)
        self.floor_slider.setValue(-1)

        # self.slider.setSingleStep(100)
        self.slider.valueChanged.connect(self.slider_changed)
        self.radius_slider.valueChanged.connect(self.radius_slider_changed)
        self.floor_slider.valueChanged.connect(self.floor_slider_changed)

        vlayout.addWidget(render_tb)

        vvlayout = QVBoxLayout()
        vvlayout.addWidget(self.plotter2.interactor)
        self.frame2.setLayout(vvlayout)

        self.tab = QTabWidget()

        tb = QToolBar()
        helper = QWidget()
        vbox = QVBoxLayout(helper)

        sa = QScrollArea()
        sa.setWidget(self.sketch)
        self.sketch.setMinimumWidth(int(1920 * 1.5))
        self.sketch.setMinimumHeight(int(1080 * 1.5))

        vbox.addWidget(tb)
        vbox.addWidget(sa)

        tb.addAction("Save", lambda: self.sketch.save(self.config.get("last"))).setIcon(QIcon.fromTheme("document-save"))
        tb.addAction("Clear", self.sketch.clear_points).setIcon(QIcon.fromTheme("edit-clear"))
        tb.addSeparator()
        tb.addAction("Delete last", self.sketch.delete_last).setIcon(QIcon.fromTheme("edit-undo"))
        tb.addSeparator()
        tb.addAction("Go", lambda: self.thread_run(self.do_create_pctl)).setIcon(QIcon.fromTheme("media-playback-start"))
        tb.addSeparator()

        self.scale_label = QLabel("Scale: 1 ")
        self.scale_label.setMinimumWidth(70)
        tb.addWidget(self.scale_label)
        self.scale_slider = QSlider()
        self.scale_slider.setMinimum(1)
        self.scale_slider.setMaximum(20)
        self.scale_slider.setValue(10)
        self.scale_slider.setOrientation(Qt.Horizontal)
        self.scale_slider.setMaximumWidth(200)
        self.scale_slider.valueChanged.connect(self.scale_slider_changed)
        tb.addWidget(self.scale_slider)

        tb.addSeparator()

        self.slider_label = QLabel("0.001")
        tb.addWidget(QLabel("Roughness: "))
        tb.addWidget(self.slider_label)
        tb.addWidget(self.slider)

        tb.addSeparator()

        self.radius_label = QLabel("4")
        tb.addWidget(QLabel("Radius: "))
        tb.addWidget(self.radius_label)
        tb.addWidget(self.radius_slider)

        tb.addSeparator()

        self.floor_label = QLabel("1")
        tb.addWidget(QLabel("Floor: "))
        tb.addWidget(self.floor_label)
        tb.addWidget(self.floor_slider)

        self.tab.addTab(helper, "Sketch")

        helper = QWidget()
        vbox = QVBoxLayout(helper)

        tb = QToolBar()
        tb.addAction("Render", lambda: self.thread_run(self.do_render_mesh)).setIcon(QIcon.fromTheme("media-playback-start"))
        tb.addAction("Save", self.save_mesh).setIcon(QIcon.fromTheme("document-save"))
        vbox.addWidget(tb)
        vbox.addWidget(self.frame2)

        self.tab.addTab(helper, "Mesh")

        self.setCentralWidget(self.tab)
        self.update_defaults()

        self.load_config()

        last = self.config.get("last")
        if last is not None and os.path.exists(last):
            self.sketch.load(last)
            self.setWindowTitle(last)

        self.show()

    def slider_changed(self):
        self.slider_label.setText(str(self.slider.value() / 1000) + " ")
        self.update_defaults()

    def radius_slider_changed(self):
        self.radius_label.setText(str(self.radius_slider.value()) + " ")
        self.update_defaults()

    def floor_slider_changed(self):
        self.floor_label.setText(str(self.floor_slider.value()) + " ")
        self.update_defaults()

    def scale_slider_changed(self):
        self.scale_label.setText("Ratio: {:.1f} ".format(self.scale_slider.value() / 10))
        self.update_defaults()

    def load_config(self):
        try:
            with open("config.yaml", "r") as f:
                self.config = yaml.full_load(f)
        except:
            pass

    def save_config(self):
        with open("config.yaml", "w") as f:
            yaml.dump(self.config, f)

    def save_yaml(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save PTG Document", self.config.get("last"), "PTG Files (*.ptg)")
        if dest and dest != "":
            if not dest.endswith(".ptg"):
                dest = dest + ".ptg"
            self.setWindowTitle(self.config.get("last"))
            self.sketch.save(dest)
            self.config["last"] = dest
            self.save_config()

    def new_yaml(self):
        dest, _ = QFileDialog.getSaveFileName(self, "Save PTG Document", "./tunnel.ptg", "PTG Files (*.ptg)")
        if dest and dest != "":
            if not dest.endswith(".ptg"):
                dest = dest + ".ptg"
            self.sketch.clear_points()
            self.config.set("last", dest)
            self.setWindowTitle(self.config.get("last"))
            self.sketch.save(dest)

    def load_yaml(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open PTG file", "", "PTG Files (*.ptg)")
        if file is not None and file != "":
            self.config.set("last", file)
            self.setWindowTitle(self.config.get("last"))
            self.sketch.load(file)
            self.config.save("gst.yaml")

    def edit(self):
        self.config.edit()
        self.config.save("gst.yaml")

    def show_mesh(self):
        mesh = pyvista.read(self.model_path + "mesh.obj")
        mesh = mesh.clip("x", invert=False, origin=(2, 0, 0))

        self.plotter2.remove_all_lights()
        pyvista.global_theme.color = "red"
        self.plotter2.clear()
        self.plotter2.enable_shadows()
        self.plotter2.set_background(color="g")
        self.plotter2.add_mesh(mesh, show_edges=True)

    def do_render_mesh(self):
        self.mesh_generator.compute_mesh()
        self.mesh_generator._meshing_params.fta_distance = self.floor_slider.value()
        self.mesh_generator.compute_floors()
        plot_mesh(self.plotter2, self.mesh_generator)
        self.pd.hide()

    def do_create_pctl(self):
        # mesh = mesh.clip("x", invert=False, origin=(2, 0, 0))

        trees = self.sketch.getPoints(self.scale_slider.value() / 10)
        if trees is None:
            self.pd.hide()
            return

        self.plotter2.remove_all_lights()
        pyvista.global_theme.color = "red"
        self.plotter2.clear()
        self.plotter2.enable_shadows()
        self.plotter2.set_background(color="grey")

        tunnel_network_params = TunnelNetworkParams.from_defaults()
        tunnel_network_params.min_distance_between_intersections = 30
        tunnel_network_params.collision_distance = 15
        tunnel_network = TunnelNetwork(params=tunnel_network_params, initial_node=False)

        tunnels_config = {}
        for tree in trees:
            tunnel = Tunnel()
            for node in tree.get("tree_nodes"):
                tunnel.append_node(node)
            tunnel_network.add_tunnel(tunnel)
            tunnels_config[tunnel] = tree.get("tree_config")

        plot_graph(self.plotter2, tunnel_network)
        plot_splines(self.plotter2, tunnel_network, color="r")

        np.random.seed(0)

        ptcl_gen_params = TunnelNetworkPtClGenParams.from_defaults()

        for k, v in tunnels_config.items():
            tunnel_ptcl_gen_params = TunnelPtClGenParams.from_defaults()
            tunnel_ptcl_gen_params.noise_multiplier = v.get("roughness", 0)
            tunnel_ptcl_gen_params.radius = v.get("radius", 1)
            ptcl_gen_params.pre_set_tunnel_params[k] = tunnel_ptcl_gen_params

        mesh_gen_params = TunnelNetworkMeshGenParams.from_defaults()
        mesh_gen_params.fta_distance = 1

        self.mesh_generator = TunnelNetworkMeshGenerator(
            tunnel_network,
            ptcl_gen_params=ptcl_gen_params,
            meshing_params=mesh_gen_params,
        )

        self.mesh_generator.compute_ptcl()
        plot_tunnel_ptcls(self.plotter2, self.mesh_generator, size=0.25, color="b")
        plot_intersection_ptcls(self.plotter2, self.mesh_generator, size=0.25, color="r")

        # self.mesh_generator.compute_mesh()
        # self.mesh_generator.compute_floors()
        # plot_mesh(self.plotter2, self.mesh_generator)

        self.tab.setTabEnabled(1, True)
        self.tab.setCurrentIndex(1)
        self.pd.hide()

    def save_mesh(self):
        if self.mesh_generator is None:
            return

        name, kind = QFileDialog.getSaveFileName(self, "Save Mesh", "", "OBJ Files (*.obj)")

        if name is not None and name != "":
            if not name.endswith(".obj"):
                name = name + ".obj"
            self.mesh_generator.save_mesh(name)

    def thread_run(self, func):
        self.plotter2.clear()

        class Runn(QRunnable):
            def __init__(self, callback):
                super().__init__()
                self.callback = callback

            def run(self) -> None:
                self.callback()

        self.tab.setCurrentIndex(2)
        self.pd = QProgressDialog(self)
        self.pd.setMaximum(0)
        self.pd.setCancelButton(None)
        self.pd.setLabelText("Rendering")
        self.pd.setModal(True)
        self.pd.show()

        run = Runn(func)
        QThreadPool.globalInstance().start(run)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.sketch.save("kakka2.yaml")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    app.exec_()
