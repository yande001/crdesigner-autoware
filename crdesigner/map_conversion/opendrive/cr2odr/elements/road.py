import enum
import math
import warnings
from typing import Dict, List, Union

import numpy as np
from commonroad.geometry.polyline_util import compute_polyline_lengths  # type: ignore
from commonroad.scenario.lanelet import Lanelet  # type: ignore
from commonroad_clcs.util import (
    compute_curvature_from_polyline,
    compute_pathlength_from_polyline,
)
from lxml import etree
from pyclothoids import SolveG2

from crdesigner.common.config.opendrive_config import open_drive_config
from crdesigner.map_conversion.opendrive.cr2odr.elements.sign import Sign
from crdesigner.map_conversion.opendrive.cr2odr.utils import config

LinkMap_T = Dict[int, Union[Dict[int, Dict[str, List[int]]], Dict[str, Dict[int, int]]]]


class GeometryType(enum.Enum):
    LINE = 1
    ARC = 2
    SPIRAL = 3
    NONE = 4


def compute_heading(polyline_left: np.ndarray, polyline_right: np.ndarray) -> np.array:
    """
    Computes the orientation of a given polyline travelled from initial
    to final coordinate. The orientation of the last coordinate is always
    assigned with the computed orientation of the penultimate one.

    :param polyline_left: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :param polyline_right: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Orientations of the polyline for each coordinate [rad]
    """
    orientation = []
    for i in range(len(polyline_left)):
        pt_1 = polyline_left[i]
        pt_2 = polyline_right[i]
        tmp = pt_2 - pt_1
        orient = np.arctan2(tmp[1], tmp[0])
        orientation.append(orient + math.pi * 0.5)

    return np.array(orientation)


def _dedupe_polyline(polyline: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Drop consecutive duplicate vertices (zero-length segments break tangents)."""
    keep = [0]
    for i in range(1, len(polyline)):
        if np.linalg.norm(polyline[i] - polyline[keep[-1]]) > eps:
            keep.append(i)
    return np.asarray(polyline)[keep]


def compute_tangent_heading(polyline: np.ndarray) -> np.ndarray:
    """
    Heading of the polyline tangent at every vertex (central differences, one-sided at the
    ends). Unlike compute_heading this does not depend on a second boundary.

    :param polyline: Polyline with 2D points [[x_0, y_0], [x_1, y_1], ...]
    :return: Orientations for each coordinate [rad]
    """
    pts = np.asarray(polyline, dtype=float)
    if len(pts) < 2:
        return np.zeros(len(pts))
    tangents = np.empty_like(pts)
    tangents[0] = pts[1] - pts[0]
    tangents[-1] = pts[-1] - pts[-2]
    if len(pts) > 2:
        tangents[1:-1] = pts[2:] - pts[:-2]
    return np.arctan2(tangents[:, 1], tangents[:, 0])


def _hermite_param_poly3(p0, p1, hdg0, hdg1):
    """
    Cubic Hermite segment from p0 (tangent direction hdg0) to p1 (tangent direction hdg1)
    expressed as OpenDRIVE paramPoly3 coefficients in the local frame of p0/hdg0 with a
    normalized parameter, plus its arc length.

    Tangent magnitudes are set to the chord length, the usual choice that keeps the
    curve free of loops for the gentle per-segment turns of a resampled polyline.

    :return: ((aU, bU, cU, dU), (aV, bV, cV, dV), length)
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    chord = float(np.linalg.norm(p1 - p0))
    m0 = chord * np.array([np.cos(hdg0), np.sin(hdg0)])
    m1 = chord * np.array([np.cos(hdg1), np.sin(hdg1)])
    d = p1 - p0
    # p(t) = p0 + m0 t + (3d - 2m0 - m1) t^2 + (-2d + m0 + m1) t^3
    c2 = 3.0 * d - 2.0 * m0 - m1
    c3 = -2.0 * d + m0 + m1
    rot = np.array([[np.cos(-hdg0), -np.sin(-hdg0)], [np.sin(-hdg0), np.cos(-hdg0)]])
    b = rot @ m0
    c = rot @ c2
    dd = rot @ c3
    # arc length by fine sampling of the polynomial
    t = np.linspace(0.0, 1.0, 33)
    xy = np.outer(t, b) + np.outer(t**2, c) + np.outer(t**3, dd)
    length = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
    return (0.0, float(b[0]), float(c[0]), float(dd[0])), (0.0, float(b[1]), float(c[1]), float(dd[1])), length


class Road:
    """
    This class adds road child element to OpenDRIVE root element
    and converts CommonRoad lanelets to OpenDRIVE roads.
    """

    counting = open_drive_config.initial_road_counting
    roads: Dict = {}
    cr_id_to_od: Dict = {}
    lanelet_to_lane: Dict = {}

    lane_2_lane_link: Dict[int, Dict[str, Dict[int, List[int]]]] = {}

    link_map: LinkMap_T = {}

    def __init__(
        self, lane_list: List[Lanelet], number_of_lanes: int, root: etree.Element, junction_id: int
    ) -> None:
        """
        This function let class road to initialize the object with lane_list, number_of_lanes, root etree element,
        current lanelet, junction_id and converts the CommonRoad roads into OpenDRIVE roads.

        :param lane_list: list of lanelets
        :param number_of_lanes: number of lanes on the road
        :param root: OpenDRIVE etree element
        :param junction_id: id of junction
        """
        Road.counting += 1
        Road.roads[Road.counting] = self
        Road.lane_2_lane_link[Road.counting] = {config.SUCC_TAG: {}, config.PRED_TAG: {}}
        self.junction_id = junction_id

        # contains etree elements for lanelinks
        self.lane_link: dict = {}

        self.links: dict = {}
        self.inner_links: dict = {}
        for lane in lane_list:
            self.links[lane.lanelet_id] = {
                config.SUCC_TAG: lane.successor,
                config.PRED_TAG: lane.predecessor,
            }
        Road.link_map[Road.counting] = self.links

        # determine center lane by finding where driving direction changes
        self.lane_list = lane_list
        self.number_of_lanes = number_of_lanes

        i = 0
        while i < self.number_of_lanes and lane_list[i].adj_left_same_direction:
            i += 1

        self.center_number = i

        self.plan_view_mode = open_drive_config.cr2odr_plan_view
        if self.plan_view_mode == "param_poly3":
            # reference line = the raw boundary polyline; headings are its tangents
            self.center = _dedupe_polyline(self.lane_list[i].left_vertices[:, :2])
            self.hdg = compute_tangent_heading(self.center)
        else:
            self.center = self.lane_list[i].left_vertices
            self.hdg = compute_heading(self.center, self.lane_list[i].center_vertices)
            self.center = (
                np.insert(
                    self.center,
                    self.center.shape[0] - 1,
                    np.array(
                        [
                            self.center[-1][0] - np.cos(self.hdg[-1]) * 0.01,
                            self.center[-1][1] - np.sin(self.hdg[-1]) * 0.01,
                        ]
                    ),
                    0,
                )
                if self.hdg[-1] != 0.0
                else self.center
            )
            self.hdg = (
                np.insert(self.hdg, self.hdg.shape[0] - 1, self.hdg[-1])
                if self.hdg[-1] != 0.0
                else self.hdg
            )

        for i in range(0, number_of_lanes):
            Road.cr_id_to_od[lane_list[i].lanelet_id] = Road.counting

        self.root = root
        self.road = etree.SubElement(root, config.ROAD_TAG)

        self.link = self.set_child_of_road(config.LINK_TAG)

        self.type = etree.SubElement(self.road, config.TYPE_TAG)
        self.type.set(config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))
        self.type.set(config.TYPE_TAG, config.TOWN_TAG)

        # planView - here goes all the geometry stuff
        self.plan_view = self.set_child_of_road(config.PLAN_VIEW_TAG)
        if self.plan_view_mode == "param_poly3":
            length = self.set_plan_view_param_poly3()
        else:
            length = self.set_plan_view()
        self.length = length

        self.elevation_profile = self.set_child_of_road(config.ELEVATION_PROFILE_TAG)
        self.lateral_profile = self.set_child_of_road(config.LATERAL_PROFILE_TAG)
        self.lanes = self.set_child_of_road(config.LANES_TAG)
        self.lane_sections()

        # objects contain static obstacles
        self.objects = self.set_child_of_road(config.OBJECTS_TAG)

        # signals contains traffic signs and traffic lights
        self.signals = self.set_child_of_road(config.SIGNALS_TAG)

        self.road.set(config.NAME_TAG, "")
        self.road.set(config.LENGTH_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, length))

        self.road.set(config.ID_TAG, str.format(config.ID_FORMAT_PATTERN, Road.counting))

        self.road.set(config.JUNCTION_TAG, str(junction_id))

        # add lane indices to links
        self.links[config.LANE_INDICES_TAG] = self.inner_links

    def add_junction_linkage(self, element_id: int, relation: str) -> None:
        """
        This function adds relation(successor/predecessor) child element to link parent element.

        :param element_id: element id
        :param relation: successor/predecessor
        """
        self.element_type = etree.SubElement(self.link, relation)
        self.element_type.set(config.ELEMENT_ID_TAG, str(element_id))
        self.element_type.set(config.ELEMENT_TYPE_TAG, config.JUNCTION_TAG)
        if relation == config.SUCCESSOR_TAG:
            self.element_type.set(config.CONTACT_POINT_TAG, config.START_TAG)
        elif relation == config.PREDECESSOR_TAG:
            self.element_type.set(config.CONTACT_POINT_TAG, config.END_TAG)
        else:
            raise ValueError("Relation must be either successor or predecessor")

    def add_simple_linkage(
        self,
        links: Dict[str, List[int]],
        len_succ: int,
        len_pred: int,
        lane_2_lane: Dict[str, Dict[int, List[int]]],
    ) -> None:
        """
        This function add successor/predecessor child element to link parent element and
        each successor/predecessor are linked with its corresponding landLink id.
        This happens when a road has exactly one successor/predecessor.

        :param links: A dictionary with successor/predecessor as key and list of road linkage id as value
        :param len_succ: Number of successors
        :param len_pred: Number of predecessors
        such as mergeLinkage, roadLinkage, laneIndices
        :param lane_2_lane: Dictionary with successor/predecessor as key and dictionaries of corresponding ids as value
        """
        if len_succ == 1:
            successor = self.element_type = etree.SubElement(self.link, config.SUCCESSOR_TAG)
            successor.set(config.ELEMENT_TYPE_TAG, config.ROAD_TAG)
            successor.set(config.ELEMENT_ID_TAG, str(links[config.SUCC_TAG][0]))
            successor.set(config.CONTACT_POINT_TAG, config.START_TAG)

            # lane_2_lane linkage
            for lane_id, successors in lane_2_lane[config.SUCC_TAG].items():
                for succ_id in successors:
                    if self.lane_link.get(lane_id) is not None:
                        # if condition added to prevent failure -> TODO check why sometime lane_id does not exist
                        succ = etree.SubElement(self.lane_link[lane_id], config.SUCCESSOR_TAG)
                        succ.set(config.ID_TAG, str(succ_id))

        if len_pred == 1:
            predecessor = self.element_type = etree.SubElement(self.link, config.PREDECESSOR_TAG)
            predecessor.set(config.ELEMENT_TYPE_TAG, config.ROAD_TAG)
            predecessor.set(config.ELEMENT_ID_TAG, str(links[config.PRED_TAG][0]))
            predecessor.set(config.CONTACT_POINT_TAG, config.END_TAG)

            # lane_2_lane linkage
            for lane_id, predecessors in lane_2_lane[config.PRED_TAG].items():
                for pred_id in predecessors:
                    pred = etree.SubElement(self.lane_link[lane_id], config.PREDECESSOR_TAG)
                    pred.set(config.ID_TAG, str(pred_id))

    def set_child_of_road(self, name: str) -> etree:
        """
        This function creates child element(name) and add it to road parent element.

        :param name: An elment name to be added to road parent element.
        :return: An etree road element with added child element.
        """
        return etree.SubElement(self.road, name)

    def set_plan_view(self) -> float:
        """
        This function compute geometric elements required for planview.
        Geometric elements such as line, spiral, curve, arc length are computed.

        :return: Length of lanelet
        """
        curv = compute_curvature_from_polyline(self.center)
        arc_length = compute_pathlength_from_polyline(self.center)
        curv_dif = np.ediff1d(curv)
        # loop through all the points in the polyline check if
        # the delta curvature is below DEVIAT
        # could be more smooth, if needed, with resampling with a
        # smaller step size

        cur_idx = 1
        last_idx = 0
        cur_type = GeometryType.NONE
        switch = GeometryType.NONE

        while cur_idx <= len(self.center) - 1:
            # previously line or start and no orientation change
            if np.isclose(
                self.hdg[cur_idx - 1], self.hdg[cur_idx], open_drive_config.heading_threshold
            ) and cur_type in [
                GeometryType.LINE,
                GeometryType.NONE,
            ]:
                cur_type = GeometryType.LINE
            # previously arc or start and curvature stays constant
            elif np.isclose(
                curv[last_idx], curv[cur_idx], open_drive_config.curvature_threshold
            ) and cur_type in [
                GeometryType.ARC,
                GeometryType.NONE,
            ]:
                cur_type = GeometryType.ARC
            # start and curvature dif increased or previously spiral and curvature dif stayed constant
            elif (
                cur_idx == 1
                and not np.isclose(
                    curv_dif[cur_idx - 1],
                    curv_dif[cur_idx - 2],
                    open_drive_config.curvature_dif_threshold,
                )
                or cur_idx > 1
                and np.isclose(
                    curv_dif[cur_idx - 1],
                    curv_dif[cur_idx - 2],
                    open_drive_config.curvature_dif_threshold,
                )
                and cur_type in [GeometryType.SPIRAL, GeometryType.NONE]
            ):
                cur_type = GeometryType.SPIRAL
            # previously no line and orientation does not change -> switch to line
            elif np.isclose(
                self.hdg[cur_idx - 1], self.hdg[cur_idx], open_drive_config.heading_threshold
            ) and cur_type not in [
                GeometryType.LINE,
                GeometryType.NONE,
            ]:
                switch = GeometryType.LINE
            # previously no arc and curvature stays constant -> switch to arc
            elif np.isclose(
                curv[last_idx], curv[cur_idx], open_drive_config.curvature_threshold
            ) and cur_type not in [
                GeometryType.ARC,
                GeometryType.NONE,
            ]:
                switch = GeometryType.ARC
            else:
                switch = GeometryType.SPIRAL
            # we have to change the geometry type -> define arc/clothoid/line
            if switch != GeometryType.NONE:
                self.print_geometry(arc_length, cur_idx - 1, cur_type, curv, last_idx)
                cur_type = switch
                switch = GeometryType.NONE
                last_idx = cur_idx - 1
            cur_idx += 1
        if switch == GeometryType.NONE:
            self.print_geometry(arc_length, cur_idx - 1, cur_type, curv, last_idx)
        return float(arc_length[-1])

    def set_plan_view_param_poly3(self) -> float:
        """
        Plan view as one G1-continuous cubic (paramPoly3) per reference-line segment.
        The road length is the sum of the cubic arc lengths, so <geometry s> and the
        road <length> are consistent by construction.

        :return: Length of the reference line
        """
        s = 0.0
        for i in range(len(self.center) - 1):
            u_coeffs, v_coeffs, length = _hermite_param_poly3(
                self.center[i], self.center[i + 1], self.hdg[i], self.hdg[i + 1]
            )
            if length <= 0.0:
                continue
            self.print_param_poly3(
                s,
                float(self.center[i][0]),
                float(self.center[i][1]),
                float(self.hdg[i]),
                length,
                u_coeffs,
                v_coeffs,
            )
            s += length
        return s

    def print_param_poly3(self, s, x, y, hdg, length, u_coeffs, v_coeffs) -> None:
        """
        Geometry child element with a paramPoly3 child (normalized parameter range).
        """
        geometry = etree.SubElement(self.plan_view, config.GEOMETRY_TAG)
        geometry.set(config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, s))
        geometry.set(config.GEOMETRY_X_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, x))
        geometry.set(config.GEOMETRY_Y_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, y))
        geometry.set(config.GEOMETRY_HEADING_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, hdg))
        geometry.set(config.LENGTH_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, length))

        poly = etree.SubElement(geometry, config.PARAM_POLY3_TAG)
        for name, value in zip(config.PARAM_POLY3_COEFFS, (*u_coeffs, *v_coeffs)):
            poly.set(name, str.format(config.DOUBLE_FORMAT_PATTERN, value))
        poly.set(config.PARAM_POLY3_P_RANGE_TAG, config.PARAM_POLY3_P_RANGE_NORMALIZED)

    def print_geometry(
        self,
        path_length: np.array,
        end_idx: int,
        geo_type: GeometryType,
        curv: np.array,
        start_idx: int,
    ):
        this_length = path_length[end_idx] - path_length[start_idx]
        if geo_type == GeometryType.LINE:
            self.print_line(
                path_length[start_idx],
                float(self.center[start_idx][0]),
                float(self.center[start_idx][1]),
                float(self.hdg[start_idx]),
                this_length,
            )
        elif geo_type == GeometryType.ARC and abs(curv[end_idx]) < 1e-9:
            # an arc with curvature 0 is a line; readers divide by the curvature
            self.print_line(
                path_length[start_idx],
                float(self.center[start_idx][0]),
                float(self.center[start_idx][1]),
                float(self.hdg[start_idx]),
                this_length,
            )
        elif geo_type == GeometryType.ARC:
            self.print_arc(
                path_length[start_idx],
                float(self.center[start_idx][0]),
                float(self.center[start_idx][1]),
                float(self.hdg[start_idx]),
                this_length,
                curv[end_idx],
            )
        elif geo_type == GeometryType.SPIRAL:
            clothoids = SolveG2(
                self.center[start_idx][0],
                self.center[start_idx][1],
                self.hdg[start_idx],
                curv[start_idx],
                self.center[end_idx][0],
                self.center[end_idx][1],
                self.hdg[end_idx],
                curv[end_idx],
            )
            self.print_spiral(
                path_length[start_idx],
                float(self.center[start_idx][0]),
                float(self.center[start_idx][1]),
                float(self.hdg[start_idx]),
                clothoids[0].Parameters[5],
                curv[start_idx],
                clothoids[0].KappaEnd,
            )
            self.print_spiral(
                path_length[start_idx] + clothoids[0].Parameters[5],
                clothoids[1].XStart,
                clothoids[1].YStart,
                clothoids[1].ThetaStart,
                clothoids[1].Parameters[5],
                clothoids[1].KappaStart,
                clothoids[1].KappaEnd,
            )
            self.print_spiral(
                path_length[start_idx] + clothoids[0].Parameters[5] + clothoids[1].Parameters[5],
                clothoids[2].XStart,
                clothoids[2].YStart,
                clothoids[2].ThetaStart,
                clothoids[2].Parameters[5],
                clothoids[2].KappaStart,
                curv[end_idx],
            )

    # xodr for lines
    def print_line(self, s: float, x: float, y: float, hdg: float, length: float) -> None:
        """
        This function print line on OpenDrive file.
        Geometry child element is created with corresponding attributes and added to planview parent element.
        Line child element is added to geometry parent element.

        :param s: s-coordinate of start position
        :param x: Start position (x inertial)
        :param y: Start position (y inertial)
        :param hdg: Start orientation (inertial heading)
        :param lenght: Length of the element’s reference line
        """
        geometry = etree.SubElement(self.plan_view, config.GEOMETRY_TAG)
        geometry.set(config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, s))
        geometry.set(config.GEOMETRY_X_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, x))
        geometry.set(config.GEOMETRY_Y_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, y))
        geometry.set(config.GEOMETRY_HEADING_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, hdg))
        geometry.set(config.LENGTH_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, length))

        etree.SubElement(geometry, config.LINE_TAG)

    # xodr for spirals
    def print_spiral(
        self,
        s: float,
        x: float,
        y: float,
        hdg: float,
        length: float,
        curv_start: float,
        curv_end: float,
    ) -> None:
        """
        This function print spiral on OpenDrive file.
        Geometry child element is created with corresponding attributes and added to planview parent element.
        Spiral child element is added to geometry parent element.

        :param s: s-coordinate of start-position
        :param x: Start position (x inertial)
        :param y: Start position (y inertial)
        :param hdg: Start orientation (inertial heading)
        :param length: Length of the element’s reference line
        :param curv_start: Curvature at the start of the element
        :param curv_end: Curvature at the end of the element
        """
        geometry = etree.SubElement(self.plan_view, config.GEOMETRY_TAG)
        geometry.set(config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, s))
        geometry.set(config.GEOMETRY_X_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, x))
        geometry.set(config.GEOMETRY_Y_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, y))
        geometry.set(config.GEOMETRY_HEADING_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, hdg))
        geometry.set(config.LENGTH_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, length))

        spiral = etree.SubElement(geometry, config.SPIRAL_TAG)
        spiral.set(
            config.GEOMETRY_CURV_START_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, curv_start)
        )
        spiral.set(config.GEOMETRY_CURV_END_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, curv_end))

    # xodr for arcs
    def print_arc(
        self, s: float, x: float, y: float, hdg: float, length: float, curvature: float
    ) -> None:
        """
        This function print arc on OpenDrive file.
        Geometry child element is created with corresponding attributes and added to planview parent element.
        Arc child element is added to geometry parent element.

        :param s: s-coordinate of start-position
        :param x: Start position (x inertial)
        :param y: Start position (y inertial)
        :param hdg: Start orientation (inertial heading)
        :param lenght: Length of the element’s reference line
        :param curvature: Constant curvature throughout the element
        """
        geometry = etree.SubElement(self.plan_view, config.GEOMETRY_TAG)
        geometry.set(config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, s))
        geometry.set(config.GEOMETRY_X_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, x))
        geometry.set(config.GEOMETRY_Y_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, y))
        geometry.set(config.GEOMETRY_HEADING_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, hdg))
        geometry.set(config.LENGTH_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, length))

        arc = etree.SubElement(geometry, config.ARC_TAG)
        arc.set(config.GEOMETRY_CURVATURE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, curvature))

    def print_signal(self, sig: Sign) -> None:
        """
        This function print traffic sign on OpenDrive file.
        Signal child element is created with corresponding attributes and added to road parent element.

        :param sig: Traffic sign
        """
        signal = etree.SubElement(self.signals, config.SIGNAL_TAG)
        signal.set(
            config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, sig.s)
        )
        signal.set(config.SIGNAL_T_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, sig.t))
        signal.set(config.ID_TAG, sig.id)
        signal.set(config.NAME_TAG, sig.name)
        signal.set(config.SIGNAL_DYNAMIC_TAG, sig.dynamic)
        signal.set(config.SIGNAL_ORIENTATION_TAG, sig.orientation)
        signal.set(config.SIGNAL_ZOFFSET_TAG, sig.zOffset)
        signal.set(config.SIGNAL_COUNTRY_TAG, sig.country)
        signal.set(config.TYPE_TAG, sig.type)
        signal.set(config.SIGNAL_SUBTYPE_TAG, sig.subtype)
        signal.set(config.SIGNAL_COUNTRY_REVISION_TAG, sig.country_revision)
        signal.set(config.SIGNAL_VALUE_TAG, sig.value)
        signal.set(config.SIGNAL_UNIT_TAG, sig.unit)
        signal.set(config.SIGNAL_WIDTH_TAG, sig.width)
        signal.set(config.SIGNAL_HEIGHT_TAG, sig.height)
        signal.set(config.SIGNAL_HOFFSET_TAG, sig.hOffset)

    def print_signal_ref(self, sig_ref: Sign) -> None:
        """
        This function print signal reference on OpenDrive file.
        Signal reference child element is created with corresponding attributes and added to road parent element.

        :param sig_ref: Traffic sign reference
        """
        signal_ref = etree.SubElement(self.signals, config.SIGNAL_REFERENCE_TAG)
        signal_ref.set(
            config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, sig_ref.s)
        )
        signal_ref.set(config.SIGNAL_T_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, sig_ref.t))
        signal_ref.set(config.ID_TAG, sig_ref.id)
        signal_ref.set(config.SIGNAL_ORIENTATION_TAG, sig_ref.orientation)

    def lane_sections(self) -> None:
        """
        This function add laneSection child element to road parent element and
        left, center (width 0), right elements are added to laneSection.
        Every road node in xodr contains also a lanes node with 1 or
        More lane_sections: left, center (width 0), right
        """
        section = etree.SubElement(self.lanes, config.LANE_SECTION_TAG)
        # i guess this s should not be hardcoded
        section.set(config.GEOMETRY_S_COORDINATE_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))

        center = etree.SubElement(section, config.LANE_SECTION_CENTER_TAG)
        self.lane_help(0, config.LANE_SECTION_DRIVING_TAG, 0, center, [], np.array([]))

        left = etree.SubElement(section, config.LANE_SECTION_LEFT_TAG)
        right = etree.SubElement(section, config.LANE_SECTION_RIGHT_TAG)

        # iterates through all the laneSection elements
        for i, cur in enumerate(self.lane_list):
            # calculate the width of a street
            # for some reason it looks better without resampling
            width_list = list(
                map(
                    lambda x, y: np.linalg.norm(x - y),
                    cur.right_vertices,
                    cur.left_vertices,
                )
            )

            if open_drive_config.cr2odr_piecewise_width:
                if cur is self.lane_list[self.center_number] and len(cur.left_vertices) == len(self.center):
                    # OpenDRIVE widths are measured along the road normal; pairing the two
                    # boundaries by sample index measures obliquely on tight curves
                    width_list = self._normal_widths(cur.right_vertices)
                # s of every sample measured on the *reference line*, not along the lane's
                # own centre (which is longer/shorter on curves and runs backwards for
                # opposite-direction lanes)
                width_list, dist_list = self._project_widths_on_reference(cur, width_list)
            else:
                dist_list = compute_polyline_lengths(cur.center_vertices)
            lane_id = i - self.center_number

            # lanelets to the right should get a negative id
            if lane_id <= 0:
                self.lane_help(
                    lane_id - 1, config.LANE_SECTION_DRIVING_TAG, 0, right, width_list, dist_list
                )
                Road.lanelet_to_lane[self.lane_list[i].lanelet_id] = lane_id - 1
                self.inner_links[self.lane_list[i].lanelet_id] = lane_id - 1

            # lanelets to the left should get a positive id -> opposite driving direction
            else:
                self.lane_help(
                    lane_id, config.LANE_SECTION_DRIVING_TAG, 0, left, width_list, dist_list
                )
                Road.lanelet_to_lane[self.lane_list[i].lanelet_id] = lane_id
                self.inner_links[self.lane_list[i].lanelet_id] = lane_id

    def _normal_widths(self, other: np.ndarray) -> List[float]:
        """
        Distance from every reference-line vertex to polyline ``other`` along the local
        normal (ray cast both ways, nearest hit); falls back to the nearest-point distance
        where the ray misses.
        """
        ref = np.asarray(self.center, dtype=float)[:, :2]
        other = np.asarray(other, dtype=float)[:, :2]
        seg = np.diff(other, axis=0)
        seg_len2 = np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-12)
        widths = []
        for k, p in enumerate(ref):
            normal = np.array([-np.sin(self.hdg[k]), np.cos(self.hdg[k])])
            # solve p + u*normal = a + v*seg  for each segment
            denom = normal[0] * (-seg[:, 1]) - normal[1] * (-seg[:, 0])
            rel = other[:-1] - p
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (rel[:, 0] * (-seg[:, 1]) - rel[:, 1] * (-seg[:, 0])) / denom
                v = (normal[0] * rel[:, 1] - normal[1] * rel[:, 0]) / denom
            hit = np.isfinite(u) & (v >= -1e-9) & (v <= 1 + 1e-9)
            if np.any(hit):
                widths.append(float(np.min(np.abs(u[hit]))))
            else:
                t = np.clip(np.einsum("ij,ij->i", -rel, seg) / seg_len2, 0.0, 1.0)
                proj = other[:-1] + seg * t[:, None]
                widths.append(float(np.min(np.hypot(*(p - proj).T))))
        return widths

    def _project_widths_on_reference(self, lane: Lanelet, width_list: List[float]):
        """
        Pair each lane width sample with the reference-line s of its centre vertex.
        Returns (widths, s_values) sorted by s, clipped to [0, road length], starting at 0.
        """
        ref = np.asarray(self.center, dtype=float)[:, :2]
        seg = np.diff(ref, axis=0)
        seg_len = np.hypot(seg[:, 0], seg[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        scale = self.length / cum[-1] if cum[-1] > 0 else 1.0  # polyline vs. cubic length
        s_vals = []
        ref_lane = self.lane_list[self.center_number]
        if lane is ref_lane and len(ref_lane.left_vertices) == len(width_list) == len(ref):
            # the reference line *is* this lane's left boundary: sample k sits at s_k exactly
            # (nearest-point projection is ambiguous on tight curves such as U-turns)
            s_vals = [float(c * scale) for c in cum]
        else:
            for pt in np.asarray(lane.center_vertices, dtype=float)[:, :2]:
                rel = pt - ref[:-1]
                t = np.clip(np.einsum("ij,ij->i", rel, seg) / np.maximum(seg_len**2, 1e-12), 0.0, 1.0)
                proj = ref[:-1] + seg * t[:, None]
                k = int(np.argmin(np.hypot(*(pt - proj).T)))
                s_vals.append(float((cum[k] + t[k] * seg_len[k]) * scale))
        pairs = sorted(zip(s_vals, width_list))
        # strictly increasing s, inside the road
        out_s, out_w = [], []
        for s_val, w in pairs:
            s_val = min(max(s_val, 0.0), self.length)
            if out_s and s_val <= out_s[-1] + 1e-6:
                continue
            out_s.append(s_val)
            out_w.append(float(w))
        if out_s:
            out_s[0] = 0.0
        return out_w, np.asarray(out_s)

    # nice idea to reuse the subelement generation for left center and right
    # produces something like:
    # <lane id="1" type=config.LANE_SECTION_DRIVING_TAG level=config.FALSE>
    #     <link/>
    #     <width sOffset="0.0000000000000000e+00" a="3.4996264749930002e+00"
    #       b="0.0000000000000000e+00" c="0.0000000000000000e+00" d="0.0000000000000000e+00"/>
    #     <roadMark sOffset="0.0000000000000000e+00" type=config.SOLID weight=config.STANDARD
    #       color=config.STANDARD width="1.3000000000000000e-01"/>
    # </lane>

    def lane_help(
        self,
        lane_id: int,
        lane_type: str,
        level: int,
        pos: etree._Element,
        width_list: List[Lanelet],
        dist_list: np.ndarray,
    ) -> None:
        """
        This function add lane child element to parent element which may be right, left or center.
        Link, width, roadMark elements are also added to lane element.

        :param: lane_id
        :param: Type
        :param: Level
        :param: Etree element that can be right, left or center
        :param: Width of a street
        :param: Path length of the polyline
        """
        lane_pos = etree.SubElement(pos, config.LANE_TAG)
        lane_pos.set(config.ID_TAG, str.format(config.ID_FORMAT_PATTERN, lane_id))
        lane_pos.set(config.TYPE_TAG, lane_type)
        lane_pos.set(config.LEVEL_TAG, config.FALSE)
        lane_link = etree.SubElement(lane_pos, config.LINK_TAG)
        self.lane_link[lane_id] = lane_link

        b = 0
        a = 0

        for w in width_list:
            w += open_drive_config.lane_evaluation_step

        # just do it the good ol' way
        if len(width_list) > 1:
            # just trying another method:
            x = [dist_list[0], dist_list[-1]]

            b, a = np.polyfit(x, [width_list[0], width_list[-1]], 1)

        if lane_id != 0 and open_drive_config.cr2odr_piecewise_width and len(width_list) > 1:
            # one linear piece per sample pair: follows width changes along the lane
            # (dist_list is the lane's own centre-line arc length, used as s offset)
            for k in range(len(width_list) - 1):
                ds = float(dist_list[k + 1] - dist_list[k])
                if ds <= 0.0:
                    continue
                slope = (width_list[k + 1] - width_list[k]) / ds
                width = etree.SubElement(lane_pos, config.SIGNAL_WIDTH_TAG)
                width.set(config.LANE_SOFFSET_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, dist_list[k]))
                width.set(config.LANE_A_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, width_list[k]))
                width.set(config.LANE_B_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, slope))
                width.set(config.LANE_C_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))
                width.set(config.LANE_D_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))
        elif lane_id != 0:
            # this should maybe not be hardcoded
            width = etree.SubElement(lane_pos, config.SIGNAL_WIDTH_TAG)
            width.set(config.LANE_SOFFSET_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))
            width.set(config.LANE_A_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, a))
            width.set(config.LANE_B_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, b))
            width.set(config.LANE_C_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))
            width.set(config.LANE_D_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))

        roadmark = etree.SubElement(lane_pos, config.ROAD_MARK_TAG)
        # this should maybe not be hardcoded
        roadmark.set(config.LANE_SOFFSET_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0))
        roadmark.set(config.TYPE_TAG, config.SOLID)
        roadmark.set(config.ROAD_MARK_WEIGHT_TAG, config.STANDARD)
        roadmark.set(config.ROAD_MARK_COLOR_TAG, config.STANDARD)
        roadmark.set(config.SIGNAL_WIDTH_TAG, str.format(config.DOUBLE_FORMAT_PATTERN, 0.13))
