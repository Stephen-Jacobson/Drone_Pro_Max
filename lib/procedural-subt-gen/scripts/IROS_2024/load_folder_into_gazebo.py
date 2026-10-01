import rospy
from gazebo_msgs.srv._SpawnModel import SpawnModel, SpawnModelRequest, SpawnModelResponse
from gazebo_msgs.srv._DeleteModel import DeleteModel, DeleteModelRequest, DeleteModelResponse
import os
import threading
import time

folder = "/home/lorenzo/Documents/conferences/IROS_2024/presentacion/data/different_tunnels"

base_xml = """<?xml version="1.0"?>
<sdf version="1.6">
    <model name="tunnel_network">
        <static>true</static>
        <link name="link">
            <pose>0 0 0 0 0 0</pose>
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>{mesh}</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name="visual">
                <geometry>
                    <mesh>
                        <uri>{mesh}</uri>
                    </mesh>
                </geometry>
            </visual>
        </link>
    </model>
</sdf>"""


def do_the_thing_target():
    time.sleep(2)
    for fn in os.listdir(folder):
        if fn.endswith(".dae"):
            proxy = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
            proxy.call(DeleteModelRequest("cave"))
            proxy.call(DeleteModelRequest("ground_plane"))
            time.sleep(1)
            path_to_mesh = os.path.join(folder, fn)
            proxy = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            request = SpawnModelRequest()
            request.model_xml = base_xml.format(mesh=path_to_mesh)
            request.model_name = "cave"
            request.reference_frame = ""
            print(proxy.call(request))
            input()


rospy.init_node("hola")
thread = threading.Thread(target=do_the_thing_target)
thread.start()
rospy.spin()
thread.join()
