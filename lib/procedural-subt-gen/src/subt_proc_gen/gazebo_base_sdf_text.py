WORLD_BASE_TEXT = """<sdf version='1.7'>
    <world name='default'>
        <gravity>0 0 -9.8</gravity>
        <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
        <atmosphere type='adiabatic'/>
        <physics type='ode'>
            <max_step_size>0.001</max_step_size>
            <real_time_factor>1</real_time_factor>
            <real_time_update_rate>1000</real_time_update_rate>
        </physics>
        <scene>
            <ambient>0.4 0.4 0.4 1</ambient>
            <background>0.7 0.7 0.7 1</background>
            <shadows>1</shadows>
        </scene>
        <gui fullscreen='0'>
            <camera name='user_camera'>
                <pose>-61.0486 -30.7792 27.4495 0 0.349797 0.428191</pose>
                <view_controller>orbit</view_controller>
                <projection_type>perspective</projection_type>
            </camera>
        </gui>
        {}
    </world>
</sdf>
"""

MODEL_BASE_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
    <model name="{}">
        <static>true</static>
        <link name="link">
            <pose>{} {} {} {} {} {}</pose>
            <collision name="collision">
                <geometry>
                    <mesh>
                        <uri>{}</uri>
                    </mesh>
                </geometry>
            </collision>
            <visual name='CaveWall_visual'>
                    <geometry>
                      <mesh>
                        <uri>{}</uri>
                      </mesh>
                    </geometry>
                    {} 
            </visual> 
        </link>
    </model>
</sdf>"""

MATERIAL_TEXT = """
                <material>
                  <diffuse>1 1 1 1</diffuse>
                  <specular>1 1 1 1</specular>
                  <pbr>
                    <metal>
                      <albedo_map>https://fuel.ignitionrobotics.org/1.0/OpenRobotics/models/Cave Straight Type A/tip/files/materials/textures/CaveWall_Albedo.jpg</albedo_map>
                      <normal_map>https://fuel.ignitionrobotics.org/1.0/OpenRobotics/models/Cave Straight Type A/tip/files/materials/textures/CaveWall_Normal.jpg</normal_map>
                      <roughness_map>https://fuel.ignitionrobotics.org/1.0/OpenRobotics/models/Cave Straight Type A/tip/files/materials/textures/CaveWall_Roughness.jpg</roughness_map>
                    </metal>
                  </pbr>
                  <script>
                    <uri>file:///home/lorenzo/.ignition/fuel/fuel.ignitionrobotics.org/openrobotics/models/cave%20starting%20area%20type%20b/7/materials/scripts/</uri>
                    <uri>materials/textures/</uri>
                    <name>CaveTile/CaveWall_Diffuse</name>
                  </script>
                </material>
                """

WOLD_IMPORT_BASE_TEXT = """
        <include>
            <uri>{}</uri>
            <pose>{}</pose>
        </include>
"""

BASE_LIGHT_TEXT = """
    <light name='{}' type='{}'>
      <pose>{} {} {} 0 0 0</pose>
      <direction>{} {} {}</direction>
      <diffuse>{} {} {} 1</diffuse>
      <specular>{} {} {} 1</specular>
      <attenuation>
        <range>{}</range>
        <constant>{}</constant>
        <linear>{}</linear>
        <quadratic>{}</quadratic>
      </attenuation>
      <cast_shadows>{}</cast_shadows>
    </light>
"""

ROCK = """
    <include>
        <uri>https://fuel.gazebosim.org/1.0/OpenRobotics/models/Falling Rock 1</uri>
        <pose>{}</pose>
        <static>true</static>
    </include>
"""
