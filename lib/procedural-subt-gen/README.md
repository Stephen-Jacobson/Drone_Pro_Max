# Procedural Environments for Underground Robotics

## Overview

This repo contains a Python package designed to generate underground tunnel networks in a procedural way, with the purpose of robotics testing.

## Instalation

After cloning the repository, move to it and run:

```bash
pip3 install -r requirements.txt
pip3 install -e .
```

## GUI

The GUI is implemented as a script that can be found in the folder ```scripts``` by the name of ```generate-tunnel-qt.py```.

To create an underground environment with the gui, there are three main steps.

1. Generate a sketch of the tunnel network.
2. Generate the pointcloud over the sketch by pressing the 'play' button on the sketching tab.
3. Once the pointcloud is satisfactory, generate the mesh by pressing the 'play' button on the visualization tab.

![gui_1](README/gui_1.png)
![gui_2](README/gui_2.png)
![gui_3](README/gui_3.png)

## Examples

In the folder ```scripts``` there are some executable python files, each showcasing a part of the library.

### snippet_1.py

This file showcases the different ways of creating ```Tunnel``` instances. There are three:

1. From Nodes: The user specifies the position of the nodes manually.
2. Connector tunnel: The user specifies an initial and final node and a tunnel is generated between them.
3. Grown Tunnel: The user speciefies an initial node and direction, and a tunnel is generated from it.

![Result of snippet_1.py](README/snippet1.png)

### snippet_2.py

This file shows how to use the tunnel creation functionalities inside the TunnelNetwork class.

1. ```TunnelNetwork.add_random_grown_tunnel()```: This function automates the selection of the initial node, initial direction and the checking process.
2. ```TunnelNetwork.add_random_connector()```: This function automates the seleciton of the initial and final nodes, as well as the checks of the final tunnel.

If you need to create complex and random environments, these functions are the easiest ones to use. The following images are some of the resulting tunnel networks generated with the script.

![Result of snippet_2.py](README/snippet2_1.png)
![Result of snippet_2.py](README/snippet2_2.png)

### snippet_3.py

This file shows the details of the assignment of parameters for the pointclud generation and mesh generation processes by:

1. Setting parameters for each tunnel.
2. Determining how parameters shoud be assigned for tunnels without pre-set parameters.

![Result of snippet_2.py](README/snippet3_1.png)
