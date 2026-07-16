"""This file defines some classes to serialize and load information about generated worlds in external packages"""

from subt_proc_gen.mesh_generation import TunnelNetworkMeshGenerator, Tunnel, TunnelNetwork
import numpy as np
import inspect
import json


class TunnelInfo:
    """Contains info on the axis points"""

    @classmethod
    def from_tunnel_network_mesh_generator(cls, tnmg: TunnelNetworkMeshGenerator, spline_res):
        tunnels_info = []
        for tunnel in tnmg.tunnels:
            assert isinstance(tunnel, Tunnel)
            _, aps, avs = tunnel.spline.discretize(spline_res)
            length = tunnel.spline.metric_length.item(0)
            apvs = np.hstack([aps, avs]).tolist()
            radius = tnmg.ptcl_params_of_tunnel(tunnel).radius
            tunnels_info.append(cls(tunnel._id, apvs, radius, length))
        return tunnels_info

    @classmethod
    def fromdict(cls, idict):
        instance = cls()
        for key in idict.keys():
            instance.__setattr__(key, idict[key])
        return instance

    def __init__(self, tunnel_id=None, apvs=None, radius=None, length=None):
        self.id = tunnel_id
        self.apvs = apvs
        self.radius = radius
        self.length = length

    def asdict(self):
        members = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        members = [a for a in members if not (a[0].startswith("__") and a[0].endswith("__"))]
        data = dict()
        for member in members:
            data[member[0]] = member[1]
        return data


class IntersectionInfo:
    @classmethod
    def from_TunnelNetworkMeshGenerator(cls, tnmg: TunnelNetworkMeshGenerator):
        intersections_info = []
        for intersection in tnmg._tunnel_network.intersections:
            _id = intersection.id
            position = np.array(intersection.xyz).tolist()
            tunnels = [tunnel._id for tunnel in tnmg._tunnel_network._tunnels_of_node[intersection]]
            intersections_info.append(cls(_id, position, tunnels))
        return intersections_info

    @classmethod
    def fromdict(cls, idict):
        instance = cls()
        for key in idict.keys():
            instance.__setattr__(key, idict[key])
        return instance

    def __init__(self, intersection_id=None, position=None, tunnels=None):
        self.id = intersection_id
        self.position = position
        self.tunnels = tunnels

    def asdict(self):
        members = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        members = [a for a in members if not (a[0].startswith("__") and a[0].endswith("__"))]
        data = dict()
        for member in members:
            data[member[0]] = member[1]
        return data


class WorldInfo:
    @classmethod
    def from_TunnelNetworkMeshGenerator(cls, tnmg: TunnelNetworkMeshGenerator, spline_res: float):
        # Create list of TunnelsInfo
        tunnels_info = TunnelInfo.from_tunnel_network_mesh_generator(tnmg, spline_res)
        intersections_info = IntersectionInfo.from_TunnelNetworkMeshGenerator(tnmg)
        fta_dist = tnmg._meshing_params.fta_distance
        return WorldInfo(tunnels_info=tunnels_info, intersections_info=intersections_info, fta_dist=fta_dist, spline_res=spline_res)

    @classmethod
    def fromdict(cls, idict:dict):
        instance = cls()
        for key in idict.keys():
            if key == "tunnels_info":
                instance.__setattr__(key,[TunnelInfo.fromdict(tdict)for tdict in idict[key]])
            elif key == "intersections_info":
                instance.__setattr__(key,[IntersectionInfo.fromdict(tdict)for tdict in idict[key]])
            else:
                instance.__setattr__(key,idict[key])
        return instance

    @classmethod
    def open_json(cls, path_to_file):
        with open(path_to_file, "r") as f:
            data = json.load(f)
        return cls.fromdict(data)

    def __init__(self, tunnels_info=None, intersections_info=None, fta_dist=None, spline_res  = None):
        self.tunnels_info = tunnels_info
        self.intersections_info = intersections_info
        self.fta_dist = fta_dist
        self.spline_res = spline_res

    def get_aps(self):
        aps = np.zeros((0,3))
        for tunnel in self.tunnels_info:
            assert isinstance(tunnel, TunnelInfo)
            aps = np.vstack([aps, np.array(tunnel.apvs)[:,:3]])
        return aps

    def get_ips(self):
        ips = np.zeros((0,3))
        for intersection in self.intersections_info:
            assert isinstance(intersection, IntersectionInfo)
            ips = np.vstack([ips, np.array(intersection.position)[:,:3]])
        return ips


    def get_apvs(self):
        apvs = np.zeros((0,6))
        for tunnel in self.tunnels_info:
            assert isinstance(tunnel, TunnelInfo)
            apvs = np.vstack([apvs, np.array(tunnel.apvs)])
        return apvs
    
    def get_apvrs(self):
        apvrs = np.zeros((0, 7))
        for tunnel_info in self.tunnels_info:
            assert isinstance(tunnel_info, TunnelInfo)
            t_apvrs = np.hstack([np.array(tunnel_info.apvs), np.ones([len(tunnel_info.apvs),1])*tunnel_info.radius])
            apvrs = np.vstack([apvrs, t_apvrs])
        return apvrs

    def get_intersection_positions(self):
        ipvs = np.zeros((0,3))
        for intersection in self.intersections_info:
            assert isinstance(intersection, IntersectionInfo)
            ipvs = np.vstack([ipvs, np.array(intersection.position)])
        return ipvs
    
    def aproximate_total_length(self):
        return self.spline_res*len(self.get_aps())


    def asdict(self):
        members = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))
        members = [a for a in members if not (a[0].startswith("__") and a[0].endswith("__"))]
        data = dict()
        for member in members:
            if isinstance(member[1], list):
                try:
                    data[member[0]] = [a.asdict() for a in member[1]]
                except:
                    data[member[0]] = member[1]
            else:
                data[member[0]] = member[1]
        return data

    def save_json(self, path_to_file):
        with open(path_to_file, "w+") as f:
            json.dump(self.asdict(), f)


def main():
    tn = TunnelNetwork()
    for i in range(2):
        tn.add_random_grown_tunnel(n_trials=10)
    for i in range(1):
        tn.add_random_connector_tunnel(n_trials=10)
    tnmg = TunnelNetworkMeshGenerator(tn)
    tnmg.compute_all()
    info1 = WorldInfo.from_TunnelNetworkMeshGenerator(tnmg,spline_res=0.1)
    info1_asdic = info1.asdict()
    info2 = WorldInfo.fromdict(info1_asdic)
    info2_asdic = info2.asdict()
    print(info1_asdic == info2_asdic)

if __name__ == "__main__":
    main()
