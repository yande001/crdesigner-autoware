import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from commonroad.common.common_lanelet import LaneletType, LineMarking
from commonroad.scenario.area import Area
from commonroad.scenario.lanelet import Lanelet
from commonroad.scenario.scenario import Location, Scenario
from commonroad.scenario.traffic_light import TrafficLight
from commonroad.scenario.traffic_sign import TrafficSign
from pyproj import CRS, Transformer

from crdesigner.common.config.general_config import GeneralConfig, general_config
from crdesigner.common.config.gui_config import lanelet2_default
from crdesigner.common.config.lanelet2_config import Lanelet2Config, lanelet2_config
from crdesigner.map_conversion.common.utils import generate_unique_id
from crdesigner.map_conversion.lanelet2.lanelet2 import (
    Multipolygon,
    Node,
    OSMLanelet,
    RegulatoryElement,
    Way,
    WayRelation,
)


# Lanelet subtypes that represent a drivable lane (get one_way + speed_limit under Autoware).
AUTOWARE_DRIVING_SUBTYPES = {"road", "road_shoulder", "bicycle_lane", "highway", "exit", "bus_lane"}

# Per-location default speed limit (km/h) used when the source has no MAX_SPEED sign.
# This is the built-in fallback; the effective values come from
# ``Lanelet2Config.autoware_default_speed_kmh`` and are user-overridable.
AUTOWARE_DEFAULT_SPEED_KMH = {"urban": 50.0, "nonurban": 60.0, "private": 30.0}

# CommonRoad stores speeds in m/s; Autoware speed_limit is km/h.
_MS_TO_KMH = 3.6


def _set_overriding_tags_for_bidirectional_users(lanelet: Lanelet, way_rel: WayRelation):
    """
    Function that extracts bidirectional users from CR lanelet and creates according overriding tags for L2 lanelet

    :param lanelet: Lanelet in CR format from which bidirectional users are extracted
    :param way_rel: WayRelation for which the tag_dict is updated with overriding tags according to extracted
    bidirectional users
    """
    for user_bidirectional in lanelet.user_bidirectional:
        user_bidirectional_value = user_bidirectional.value

        # change due to differences in cr and l2 formats
        if user_bidirectional_value == "priorityVehicle":
            user_bidirectional_value = "emergency"

        if user_bidirectional_value in lanelet2_config.supported_lanelet2_vehicles:
            way_rel.tag_dict["one_way:" + "vehicle:" + user_bidirectional_value] = "no"

        elif user_bidirectional_value == "vehicle":
            way_rel.tag_dict["one_way:vehicle"] = "no"

        elif user_bidirectional_value == "bicycle" or user_bidirectional_value == "pedestrian":
            way_rel.tag_dict["one_way:" + user_bidirectional_value] = "no"


def _line_marking_to_type_subtype_vertices(line_marking: LineMarking) -> [str, str]:
    """
    Function that converts CR line marking of left or right vertices to an L2 format

    :param line_marking: line marking
    :return: tuple of strings that represent type and subtype of way based on CR line markings in L2 format
    """
    lanelet2_type = "unknown"  # default type, empty dict for 'unknown' and 'no_marking'
    # as they don't exist in L2 format
    subtype = "unknown"  # default subtype, empty dict for 'unknown' and 'no_marking' as they don't exist in L2 format
    if line_marking is LineMarking.DASHED:
        lanelet2_type = "line_thin"
        subtype = "dashed"
    if line_marking is LineMarking.BROAD_DASHED:
        lanelet2_type = "line_thick"
        subtype = "dashed"
    if line_marking is LineMarking.SOLID:
        lanelet2_type = "line_thin"
        subtype = "solid"
    if line_marking is LineMarking.BROAD_SOLID:
        lanelet2_type = "line_thick"
        subtype = "solid"
    if line_marking is LineMarking.CURB:
        lanelet2_type = "road_border"
        subtype = "high"
    if line_marking is LineMarking.LOWERED_CURB:
        lanelet2_type = "road_border"
        subtype = "low"
    if line_marking is LineMarking.DASHED_SOLID:
        lanelet2_type = "line_thick"
        subtype = "dashed_solid"
    if line_marking is LineMarking.SOLID_DASHED:
        lanelet2_type = "line_thick"
        subtype = "solid_dashed"
    if line_marking is LineMarking.SOLID_SOLID:
        lanelet2_type = "line_thick"
        subtype = "solid_solid"
    if line_marking is LineMarking.DASHED_DASHED:
        lanelet2_type = "line_thick"
        subtype = "dashed_dashed"

    return lanelet2_type, subtype


def _combine_subtypes(first: str, second: str) -> str:
    """Combine the L2 line subtypes of two lanelets that share a boundary.

    Each CR lanelet reports its own half of the shared marking. For simple
    markings the two halves form a double line (``solid`` + ``solid`` ->
    ``solid_solid``). If either half is *already* a compound marking (e.g.
    ``solid_solid`` from a ``SOLID_SOLID`` road mark) it already describes the
    whole line, so it is used as-is rather than concatenated again — which
    previously produced the invalid ``solid_solid_solid_solid``.

    :param first: subtype of the first lanelet's boundary
    :param second: subtype of the second lanelet's boundary
    :return: a valid L2 line subtype
    """
    if "_" in first:
        return first
    if "_" in second:
        return second
    combined = first + "_" + second
    # dashed_dashed does not exist in L2 format
    if combined == "dashed_dashed":
        return "dashed"
    return combined


def _extract_and_convert_subtype_name(
    cr_subtypes: List[str], l2_subtypes: List[str]
) -> [str, bool]:
    """
    Function that extracts the most specific lanelet type and, if needed, converts its name as it could slightly differ
    between formats.

    :param cr_subtypes: list of the subtype names of the lanelet
    :param l2_subtypes: lanelet2 subtypes that are available in commonroad
    :return: name of the most specific subtype (converted if needed)
    and the boolean value that states if the subtype corresponds to one of the possible L2 format subtypes
    """
    # extracting lanelet types that exist in L2 format
    common_subtypes = list(set(cr_subtypes).intersection(l2_subtypes))
    # if there is not a single common lanelet type, return False
    if len(common_subtypes) == 0:
        return "", False

    # return the most specific lanelet type
    if "sidewalk" in common_subtypes:
        return "walkway", True
    if "crosswalk" in common_subtypes:
        return "crosswalk", True
    if "bicycleLane" in common_subtypes:
        return "bicycle_lane", True
    if "exitRamp" in common_subtypes:
        return "exit", True
    if "busLane" in common_subtypes:
        return "bus_lane", True
    if "highway" in common_subtypes or "interstate" in common_subtypes:
        return "highway", True
    if "urban" in common_subtypes or "country" in common_subtypes:
        return "road", True


def _traffic_light_bulb_colors(light) -> List[str]:
    """Bulb colors (top→bottom) for a traffic light's light_bulbs linestring.

    CARLA-derived signals leave ``light.color`` empty, so the lamps are inferred
    from the cycle states: red/yellow/green present in any state (``redYellow``
    contributes both red and yellow). Falls back to ``light.color`` then to the
    standard three-bulb layout so a light_bulbs linestring is always produced.
    """
    states = set()
    cycle = getattr(light, "traffic_light_cycle", None)
    if cycle is not None:
        for element in getattr(cycle, "cycle_elements", []) or []:
            states.add(element.state.value)
    colors = []
    if "red" in states or "redYellow" in states:
        colors.append("red")
    if "yellow" in states or "redYellow" in states:
        colors.append("yellow")
    if "green" in states:
        colors.append("green")
    if not colors and light.color:
        colors = [c.value for c in light.color]
    if not colors:
        colors = ["red", "yellow", "green"]
    return colors


def _vertices_are_equal(
    vertices1: List[np.ndarray], vertices2: List[np.ndarray], ways_are_equal_tolerance: float
) -> bool:
    """
    Checks if two list of vertices are equal up to a tolerance.

    :param vertices1: First vertices to compare.
    :param vertices2: Second vertices to compare.
    :param ways_are_equal_tolerance: value of the tolerance for which we mark ways as equal
    :return: True if every vertex in one list is nearly equal to the
        corresponding vertices at the same position in the other list.
    """
    if len(vertices1) != len(vertices2):
        return False
    diff = np.array(vertices1) - np.array(vertices2)
    # Max absolute coordinate deviation must be within tolerance. Using
    # np.max(np.abs(diff)) (not np.abs(np.max(diff))) makes the test symmetric in
    # its arguments — otherwise mixed-sign deltas make equality depend on call
    # order, so a shared boundary is deduplicated for one conversion order only
    # (e.g. RHT shared, LHT did not).
    if np.max(np.abs(diff)) < ways_are_equal_tolerance:
        return True
    return False


class CR2LaneletConverter:
    """
    Class to convert CommonRoad lanelet to the OSM representation.
    """

    def __init__(
        self, config: Lanelet2Config = lanelet2_config, cr_config: GeneralConfig = general_config
    ):
        """
        Initialization of CR2LaneletConverter

        :param config: Lanelet2 config parameters.
        """
        generate_unique_id(0)  # reset ID counter for next test case
        self._config = config
        self._cr_config = cr_config
        self.transformer = None
        self.osm = None
        self._id_count = 1
        self.first_nodes, self.last_nodes = None, None
        self.left_ways, self.right_ways = None, None
        self.lanelet_network = None
        self.origin_utm = (0, 0)

        self.scenario_translation = (0, 0)

        # Mapping from OpenDrive (road_id, section_id, lane_id) -> Lanelet2 relation ID.
        # Populated during conversion when lanelets carry an OpenDrive-derived description
        # (format "road_id.section_id.lane_id.width_id" set by ConversionLaneletNetwork).
        self.odr_to_l2_mapping: Dict[Tuple[int, int, int], int] = {}

    def _create_transformer(self, scenario: Scenario):
        """
        Creates a Transformer object for conversion from CR o Lanelet2.
        The input projection is determined from the geo transformation of the CR scenario if provided, otherwise the
        projection string from self._config is used,
        The output projection is set to "ETRF89".
        Additionally, the x and y translation from the scenario object are considered (if provided).
        """
        loc: Location = scenario.location
        proj_string_from = None
        if loc is not None and loc.geo_transformation is not None:
            geo_trans = loc.geo_transformation
            # get projection string from geo reference
            proj_string_from = geo_trans.geo_reference
            # get translation w.r.t. lat/lon location from scenario object
            self.scenario_translation = (geo_trans.x_translation, geo_trans.y_translation)
            # TODO: z rotation and scaling are currently ignored
            if geo_trans.z_rotation != 0.0 or geo_trans.scaling != 1:
                warnings.warn(
                    "<CR2LaneletConverter>: z_rotation and scaling are not considered during transformation"
                )
        if proj_string_from is None:
            proj_string_from = self._cr_config.proj_string_cr
        crs_from = CRS(proj_string_from)
        crs_to = CRS(lanelet2_default)
        self.transformer = Transformer.from_proj(crs_from, crs_to)

    @property
    def id_count(self) -> int:
        """
        Internal counter for giving IDs to the members of the OSM.
        Each call returns the count and increases it by one.

        :return: current id count.
        """
        tmp = self._id_count
        self._id_count += 1
        return tmp

    def __call__(self, scenario):
        """
        Convert a scenario to an OSM xml document.

        :param scenario: Scenario that will be used for conversion
        """
        self._create_transformer(scenario)
        self.osm = OSMLanelet()
        self.lanelet_network = scenario.lanelet_network
        self.first_nodes = {}  # saves first left and right node | dict() but with a faster execution
        self.last_nodes = {}  # saves last left and right node
        self.left_ways = {}
        self.right_ways = {}
        # endpoint-bucket index of created boundary ways for the geometric
        # sharing fallback (opposing lanes that CommonRoad did not mark adjacent).
        # key -> [(way_id, vertices, line_marking)]
        self._boundary_index = defaultdict(list)
        self.odr_to_l2_mapping = {}
        # Autoware intersection metadata (S6): connector lanelet -> turn direction,
        # and lanelet id -> way-relation id so the intersection_area post-pass can
        # attach the polygon reference back onto each member lanelet.
        self._turn_directions = self._build_turn_directions() if self._config.autoware else {}
        self._lanelet_to_way_rel = {}
        # CommonRoad traffic-sign id -> (L2 sign way id, sign name) for robust
        # right-of-way matching (replaces the fragile sign-position string compare).
        self._sign_way_index = {}
        # traffic_light way id -> light_bulbs way id (Autoware lamp linestring, vm-04-03)
        self._light_bulbs_by_tl = {}

        # set origin shift according to translation in scenario
        if self.scenario_translation[0] != 0 and self.scenario_translation[1] != 0:
            self.origin_utm = self.scenario_translation

        # convert lanelets
        for lanelet in scenario.lanelet_network.lanelets:
            self._convert_lanelet(lanelet)

        # convert traffic signs
        for traffic_sign in scenario.lanelet_network.traffic_signs:
            self._convert_traffic_sign(traffic_sign)

        # convert traffic lights
        for traffic_light in scenario.lanelet_network.traffic_lights:
            self._convert_traffic_light(traffic_light)

        # convert areas
        for area in scenario.lanelet_network.areas:
            self._convert_area(area)

        # build right_of_way regulatory elements; Autoware gets per-intersection
        # relations (signed yield/priority + right-before-left), the generic path
        # keeps the simple sign-only relation.
        if self._config.autoware:
            self._add_intersection_right_of_way()
        else:
            self._add_right_of_way_relation()

        # map the traffic lights and the referred lanelets to a 'right_of_way_relation' object
        self._add_regulatory_element_for_traffic_lights()

        # append the lane_change flag to osm ways if the autoware flag is set to True
        if self._config.autoware is True:
            self._append_lane_change_tags()
            # synthesize an intersection_area polygon per OpenDRIVE junction (vm-03-01)
            self._add_intersection_areas()
            # crosswalk_polygon + crosswalk regulatory element per crosswalk (vm-05-01)
            self._add_crosswalks()

        return self.osm.serialize_to_xml()

    def _add_regulatory_element_for_traffic_lights(self):
        """
        Add traffic light relations to lanelets in lanelet2 format
        """
        # One regulatory element is created for each lanelet that has a corresponding traffic_light element
        for idx, ll in enumerate(self.lanelet_network.lanelets):
            traffic_light_reference_list = []
            for light_id in ll.traffic_lights:
                # find the coordinates of the CR traffic light
                # and check them with traffic light in L2 format that we have created

                # if 'position' returns 2 values, *z will be empty. Else, it will be an array with remaining values
                x, y, *z = self.lanelet_network.find_traffic_light_by_id(light_id).position
                if len(z) == 0:
                    z = 0
                else:
                    z = z[0]

                lat_sign, lon_sign = self.transformer.transform(
                    self.origin_utm[0] + x, self.origin_utm[1] + y
                )
                for way in self.osm.ways:
                    if self.osm.find_way_by_id(way).tag_dict.get("type") == "traffic_light":
                        n_lon = self.osm.find_node_by_id(self.osm.find_way_by_id(way).nodes[0]).lon
                        n_lat = self.osm.find_node_by_id(self.osm.find_way_by_id(way).nodes[0]).lat

                        if str(lon_sign) == n_lon and str(lat_sign) == n_lat:
                            # found the same traffic light, now to reference it
                            traffic_light_reference_list.append(way)

            if len(ll.traffic_lights) > 0:
                # map the end of the lanelet to the ref_line
                # maybe map the stop line also? Lanelets from examples didn't have stopLines so double check if needed
                x, y = self._get_shared_last_nodes_from_other_lanelets(ll)
                if x is None or y is None:
                    x, y = self.last_nodes.get(ll.lanelet_id, (None, None))
                way_tl = Way(self.id_count, [x, y])
                self.osm.add_way(way_tl)
                way_list = [way_tl.id_]
                # link each referenced traffic_light to its light_bulbs lamps (vm-04-03)
                light_bulbs_list = [
                    self._light_bulbs_by_tl[w]
                    for w in traffic_light_reference_list
                    if w in self._light_bulbs_by_tl
                ]
                regulatory_element_id = self.id_count
                self.osm.add_regulatory_element(
                    RegulatoryElement(
                        regulatory_element_id,
                        traffic_light_reference_list,
                        ref_line=way_list,
                        light_bulbs=light_bulbs_list,
                        tag_dict={"subtype": "traffic_light", "type": "regulatory_element"},
                    )
                )
                get_way_relation_key = list(self.osm.way_relations.keys())[idx]
                way_rel = self.osm.find_way_rel_by_id(get_way_relation_key)
                reg_elements_arr = way_rel.regulatory_elements
                reg_elements_arr.append(regulatory_element_id)
                new_way_rel = WayRelation(
                    get_way_relation_key,
                    way_rel.left_way,
                    way_rel.right_way,
                    tag_dict=way_rel.tag_dict,
                    regulatory_elements=reg_elements_arr,
                )
                self.osm.add_way_relation(new_way_rel)

    def _convert_area(self, area: Area):
        """
        Converts a CommonRoad area to the Lanelet2 multipolygon.

        :param area: area to be converted.
        """
        outer_list = list()
        for border in area.border:
            nodes = self._create_nodes_from_vertices(border.border_vertices)
            type, subtype = _line_marking_to_type_subtype_vertices(border.line_marking)
            way = Way(self.id_count, nodes, {"subtype": subtype, "type": type})
            self.osm.add_way(way)
            outer_list.append(way.id_)
        tag_dict = {}
        if area.area_types:
            for area_type in area.area_types:
                tag_dict["subtype"] = str(area_type.value)
        multipolygon = Multipolygon(self.id_count, outer_list, tag_dict)
        self.osm.add_multipolygon(multipolygon)

    def _convert_traffic_light(self, light: TrafficLight):
        """
        Add traffic light to the lanelet2 format
        """
        traffic_light_id = self.id_count
        autoware = self._config.autoware

        # consider z-coordinate
        z = 5  # Set a base of 5m and replace it with new value
        if len(light.position) == 3:
            z = light.position[2]

        id1 = self.id_count

        # since 3 nodes are needed to represent the sign in the l2 format (only 1 in the cr format)
        # create another 2 nodes that are close to the first one
        # create a node that represent the sign position

        lat1, lon1 = self.transformer.transform(
            self.origin_utm[0] + light.position[0], self.origin_utm[1] + light.position[1]
        )
        lat2, lon2 = self.transformer.transform(
            self.origin_utm[0] + light.position[0] + 0.1,
            self.origin_utm[1] + light.position[1] + 0.1,
        )
        if not autoware:
            lat3, lon3 = self.transformer.transform(
                self.origin_utm[0] + light.position[0] - 0.1,
                self.origin_utm[1] + light.position[1] - 0.1,
            )
            id3 = self.id_count

        id2 = self.id_count

        # creating and adding those nodes to our osm
        localx1, localy1, localx2, localy2, localx3, localy3 = None, None, None, None, None, None

        if self._config.use_local_coordinates:
            localx1, localy1 = light.position[0], light.position[1]
            localx2, localy2 = light.position[0] - 0.1, light.position[1] + 0.1
            localx3, localy3 = light.position[0] - 0.1, light.position[1] - 0.1
        self.osm.add_node(
            Node(id1, lat1, lon1, z, autoware=autoware, local_x=localx1, local_y=localy1)
        )
        self.osm.add_node(
            Node(id2, lat2, lon2, z, autoware=autoware, local_x=localx2, local_y=localy2)
        )
        if not autoware:
            self.osm.add_node(
                Node(id3, lat3, lon3, z, autoware=autoware, local_x=localx3, local_y=localy3)
            )

        # bulb colors: light.color is empty for CARLA-derived signals, so derive
        # them from the cycle states (vm-04-03).
        bulb_colors = _traffic_light_bulb_colors(light)
        traffic_light_subtype = "_".join(c.value for c in light.color) or "_".join(bulb_colors)
        # Autoware supports that traffic lights only consist of 2 nodes instead of 3
        if autoware:
            self.osm.add_way(
                Way(
                    traffic_light_id,
                    [id1, id2],
                    tag_dict={
                        "subtype": traffic_light_subtype,
                        "type": "traffic_light",
                        "height": "1.2",
                    },
                )
            )
            # light_bulbs lamp linestring linked to this traffic_light (vm-04-03)
            self._emit_light_bulbs(light, traffic_light_id, z, bulb_colors)
        else:
            self.osm.add_way(
                Way(
                    traffic_light_id,
                    [id1, id2, id3],
                    tag_dict={"subtype": traffic_light_subtype, "type": "traffic_light"},
                )
            )

    def _emit_light_bulbs(self, light, traffic_light_id, z, bulb_colors):
        """Emit an Autoware ``light_bulbs`` lamp linestring for a traffic light
        (vm-04-03) and record it against the traffic_light way.

        OpenDRIVE/CommonRoad carries no individual lamp geometry, so the bulbs are
        synthesised at the light's position, stacked vertically (red top → green
        bottom, ~0.35 m apart), each node tagged with its ``color``. The way is
        tagged ``traffic_light_id`` so Autoware can pair lamps with their light.
        """
        lat, lon = self.transformer.transform(
            self.origin_utm[0] + light.position[0], self.origin_utm[1] + light.position[1]
        )
        local_x = local_y = None
        if self._config.use_local_coordinates:
            local_x, local_y = light.position[0], light.position[1]
        bulb_nodes = []
        for i, color in enumerate(bulb_colors):
            nid = self.id_count
            self.osm.add_node(
                Node(nid, lat, lon, z - i * 0.35, autoware=True,
                     local_x=local_x, local_y=local_y, extra_tags={"color": color})
            )
            bulb_nodes.append(nid)
        if not bulb_nodes:
            return
        bulbs_way = Way(
            self.id_count,
            bulb_nodes,
            tag_dict={"type": "light_bulbs", "traffic_light_id": str(traffic_light_id)},
        )
        self.osm.add_way(bulbs_way)
        self._light_bulbs_by_tl[str(traffic_light_id)] = bulbs_way.id_

    def _add_right_of_way_relation(self):
        """
        Add a right_of_way regulatory element from stop/yield/priority signs.

        The lanelet carrying each sign is found by the sign's CommonRoad id
        (``self._sign_way_index``, populated in ``_convert_traffic_sign``) rather
        than by string-comparing transformed coordinates, which never matched in
        practice (float formatting + an ``ele`` "0" vs "0.0" mismatch) so the
        relation came out empty even on maps with stop/yield signs (vm-03-10).
        """
        refers = []
        yield_ways = []
        right_of_ways = []
        ref_line = []
        # convert stop lines of yield lanelets to ways and add them to L2 map
        dict_stop_lines = self._create_stop_line_to_way_dictionary()

        for ll in self.lanelet_network.lanelets:
            if not ll.traffic_signs:
                continue
            way_rel_id = self._lanelet_to_way_rel.get(ll.lanelet_id)
            if way_rel_id is None:
                continue
            for traffic_sign_id in ll.traffic_signs:
                entry = self._sign_way_index.get(traffic_sign_id)
                if entry is None:
                    continue
                sign_way_id, sign_name = entry
                if sign_name == "MAX_SPEED":
                    continue  # speed limits are handled as their own reg-elem
                refers.append(sign_way_id)
                if sign_name in ("YIELD", "STOP"):
                    yield_ways.append(way_rel_id)
                    stop_line_way = dict_stop_lines.get(ll.lanelet_id)
                    if stop_line_way is not None:
                        ref_line.append(stop_line_way)
                elif sign_name in ("RIGHT_OF_WAY", "PRIORITY"):
                    right_of_ways.append(way_rel_id)

        # do not add right_of_way_rel if there are no signs
        if len(refers) > 0:
            self.osm.add_regulatory_element(
                RegulatoryElement(
                    self.id_count,
                    refers,
                    yield_ways,
                    right_of_ways,
                    ref_line=ref_line,
                    tag_dict={"subtype": "right_of_way", "type": "regulatory_element"},
                )
            )

    def _add_intersection_right_of_way(self):
        """Build right_of_way regulatory elements for junctions (vm-03-10/11).

        OpenDRIVE ``<junction><priority>`` is not parsed by odr2cr, so priority is
        inferred from the CommonRoad model via two complementary mechanisms:

        * **(A) Sign-controlled** (vm-03-10) — every lanelet carrying a STOP/YIELD
          sign yields. One relation per signed lanelet: the sign as ``refers``, the
          lanelet as ``yield``, its stop line as ``ref_line``. Priority is the cross
          traffic of the junction the lanelet feeds *when* the intersection model
          links it (in the CARLA fixtures odr2cr leaves the signs unlinked, so the
          priority side is usually empty — the stop obligation is still recorded).
        * **(B) Unsignalised right-before-left** (vm-03-11) — from ``left_of``:
          ``X.left_of == Y`` means Y sits to X's left, i.e. X is to Y's right, so
          **Y yields to X**. One relation per yielding incoming, pairing its
          connectors (yield) with the priority incoming's connectors. Approaches
          already governed by a stop/yield sign are skipped (the sign wins).

        Yield/priority members are connector (successor) lanelets, which already
        carry ``turn_direction`` / ``intersection_area``.
        """
        dict_stop_lines = self._create_stop_line_to_way_dictionary()

        def connector_rels_from_ids(ids):
            return [self._lanelet_to_way_rel[i] for i in ids if i in self._lanelet_to_way_rel]

        def connector_ids(incoming):
            return (
                set(incoming.successors_left)
                | set(incoming.successors_straight)
                | set(incoming.successors_right)
            )

        def emit(refers, yield_ways, right_of_ways, ref_line):
            if not yield_ways:
                return
            self.osm.add_regulatory_element(
                RegulatoryElement(
                    self.id_count,
                    refers,
                    yield_ways,
                    right_of_ways,
                    ref_line=ref_line,
                    tag_dict={"subtype": "right_of_way", "type": "regulatory_element"},
                )
            )

        # connector lanelet id -> (intersection, owning incoming id), for priority lookup
        connector_owner = {}
        for intersection in self.lanelet_network.intersections:
            for inc in intersection.incomings:
                for cid in connector_ids(inc):
                    connector_owner[cid] = (intersection, inc.incoming_id)

        # (A) sign-controlled approaches
        signed_lanelets = set()
        for ll in self.lanelet_network.lanelets:
            if not ll.traffic_signs:
                continue
            sign_ways = []
            for sid in ll.traffic_signs:
                entry = self._sign_way_index.get(sid)
                if entry is not None and entry[1] in ("YIELD", "STOP"):
                    sign_ways.append(entry[0])
            if not sign_ways:
                continue
            way_rel_id = self._lanelet_to_way_rel.get(ll.lanelet_id)
            if way_rel_id is None:
                continue
            signed_lanelets.add(ll.lanelet_id)
            # priority = the other incomings' connectors at the junction this feeds
            right_of_ways = []
            for succ in ll.successor:
                owner = connector_owner.get(succ)
                if owner is None:
                    continue
                intersection, my_inc = owner
                for inc in intersection.incomings:
                    if inc.incoming_id != my_inc:
                        right_of_ways += connector_rels_from_ids(connector_ids(inc))
            stop_way = dict_stop_lines.get(ll.lanelet_id)
            emit(sign_ways, [way_rel_id], right_of_ways, [stop_way] if stop_way else [])

        # (B) unsignalised right-before-left
        for intersection in self.lanelet_network.intersections:
            info = {}
            for inc in intersection.incomings:
                info[inc.incoming_id] = {
                    "connectors": connector_rels_from_ids(connector_ids(inc)),
                    "signed": any(l in signed_lanelets for l in inc.incoming_lanelets),
                    "stops": [
                        dict_stop_lines[l] for l in inc.incoming_lanelets if l in dict_stop_lines
                    ],
                }
            priorities = defaultdict(list)  # yielding incoming id -> [priority incoming ids]
            for inc in intersection.incomings:
                yielding = inc.left_of
                if yielding is None or yielding not in info:
                    continue
                if info[yielding]["signed"]:
                    continue  # a stop/yield sign already governs this approach
                priorities[yielding].append(inc.incoming_id)
            for yielding_id, prio_ids in priorities.items():
                right_of_ways = []
                for pid in prio_ids:
                    right_of_ways += info[pid]["connectors"]
                emit([], info[yielding_id]["connectors"], right_of_ways,
                     info[yielding_id]["stops"])

    def _create_stop_line_to_way_dictionary(self) -> Dict[int, str]:
        """
        Creates a dictionary that maps lanelet (CR) id's with their corresponding way (stop line) (L2) id's.

        :return: Above stated dictionary
        """
        dict_stop_lines = {}
        for ll in self.lanelet_network.lanelets:
            if ll.stop_line:
                # found a lanelet with a stop line
                stop_line = ll.stop_line
                # stop line has 2 points (each point has x and y coordinates)
                stop_line_start = stop_line.start
                stop_line_end = stop_line.end
                # transform the x and y coordinates to the L2 coordinate system
                lat_start, lon_start = self.transformer.transform(
                    self.origin_utm[0] + stop_line_start[0], self.origin_utm[1] + stop_line_start[1]
                )
                lat_end, lon_end = self.transformer.transform(
                    self.origin_utm[0] + stop_line_end[0], self.origin_utm[1] + stop_line_end[1]
                )

                # consider the z-coordinate
                z_start = 0
                if len(stop_line_start) == 3:
                    z_start = stop_line_start[2]
                z_end = 0
                if len(stop_line_end) == 3:
                    z_end = stop_line_end[2]

                # create nodes from the points and add them to the osm
                node_start = Node(
                    self.id_count, lat_start, lon_start, z_start, autoware=self._config.autoware
                )
                node_end = Node(
                    self.id_count, lat_end, lon_end, z_end, autoware=self._config.autoware
                )
                self.osm.add_node(node_start)
                self.osm.add_node(node_end)
                # create a way from newly created nodes and add it to the osm
                stop_line_way = Way(
                    self.id_count, [node_start.id_, node_end.id_], tag_dict={"type": "stop_line"}
                )
                self.osm.add_way(stop_line_way)
                # map the way with the lanelet
                dict_stop_lines[ll.lanelet_id] = stop_line_way.id_
        return dict_stop_lines

    def _convert_traffic_sign(self, sign: TrafficSign):
        """
        Convert a traffic sign to way which will be mapped by RightOfWay Regulatory element
        Add the resulting right of way relation to the OSM.

        :param sign: Traffic Sign to be converted.
        """
        # create a node that represent the sign position
        lat_1, lon_1 = self.transformer.transform(
            self.origin_utm[0] + sign.position[0], self.origin_utm[1] + sign.position[1]
        )
        id1 = self.id_count

        # since 2 nodes are needed to represent the sign in the l2 format (only 1 in the cr format)
        # create another node that is close to the first one
        lat_2, lon_2 = self.transformer.transform(
            self.origin_utm[0] + sign.position[0] + 0.25,
            self.origin_utm[1] + sign.position[1] + 0.25,
        )
        id2 = self.id_count

        # check if the sign has z-coordinate (elevation)
        z = 0
        if len(sign.position) == 3:  # the sign has a z-coordinate
            z = sign.position[2]

        # creating and adding those nodes to our osm
        self.osm.add_node(Node(id1, lat_1, lon_1, z, autoware=self._config.autoware))
        self.osm.add_node(Node(id2, lat_2, lon_2, z, autoware=self._config.autoware))

        # matching the type of the traffic sign
        sign_id = sign.traffic_sign_elements[0].traffic_sign_element_id
        val = ""
        for country in self._config.supported_countries:
            for k in country:
                if k == sign_id:
                    val = k.value

        traffic_sign_wayid = self.id_count
        virtual = sign.virtual

        # extract the country name of the sign, so we can map it to a dictionary
        sign_country_name = str(
            type(sign.traffic_sign_elements[0].traffic_sign_element_id).__name__
        )

        # map the supported countries to their 2 letter prefixs
        country_prefix_dictionary = self._config.supported_countries_prefixes

        subtype = country_prefix_dictionary[sign_country_name]

        # if it is a speed sign, don't add way but add regulatory element "speed_limit"
        self.osm.add_way(
            Way(
                traffic_sign_wayid,
                [id1, id2],
                tag_dict={
                    "subtype": subtype + str(val),
                    "type": "traffic_sign",
                    "virtual": str(virtual),
                },
            )
        )

        # index the sign by its CommonRoad id so right-of-way relations can be
        # built by id lookup (vm-03-10) instead of fragile position matching.
        self._sign_way_index[sign.traffic_sign_id] = (traffic_sign_wayid, sign_id.name)

    def _convert_lanelet(self, lanelet: Lanelet):
        """
        Convert a lanelet to a way relation.
        Add the resulting relation and its ways and nodes to the OSM.

        :param lanelet: Lanelet to be converted.
        """
        # check if there are shared ways
        right_way_id = self._get_potential_right_way(lanelet)
        left_way_id = self._get_potential_left_way(lanelet)
        left_created = left_way_id is None
        right_created = right_way_id is None

        left_nodes, right_nodes = self._create_nodes(lanelet, left_way_id, right_way_id)

        self.first_nodes[lanelet.lanelet_id] = (left_nodes[0], right_nodes[0])
        self.last_nodes[lanelet.lanelet_id] = (left_nodes[-1], right_nodes[-1])

        if not left_way_id:
            left_way = Way(self.id_count, left_nodes)
            lanelet2_type, subtype = _line_marking_to_type_subtype_vertices(
                lanelet.line_marking_left_vertices
            )
            if lanelet2_type != "unknown":
                left_way.tag_dict = {"type": lanelet2_type, "subtype": subtype}
            self.osm.add_way(left_way)
            left_way_id = left_way.id_
        if not right_way_id:
            right_way = Way(self.id_count, right_nodes)
            lanelet2_type, subtype = _line_marking_to_type_subtype_vertices(
                lanelet.line_marking_right_vertices
            )
            if lanelet2_type != "unknown":
                right_way.tag_dict = {"type": lanelet2_type, "subtype": subtype}
            self.osm.add_way(right_way)
            right_way_id = right_way.id_

        # index newly created boundaries so later lanelets can share them via the
        # geometric fallback (opposing centerlines lacking CommonRoad adjacency).
        if left_created:
            self._index_boundary(left_way_id, lanelet.left_vertices, lanelet.line_marking_left_vertices)
        if right_created:
            self._index_boundary(right_way_id, lanelet.right_vertices, lanelet.line_marking_right_vertices)

        # create a list of lanelet type values, so we can extract the most specific one to convert it to L2 format
        lanelet_types = []
        for type_enum in list(lanelet.lanelet_type):
            lanelet_types.append(type_enum.value)

        # extracting and converting the most specific lanelet type
        subtype, subtype_in = _extract_and_convert_subtype_name(
            lanelet_types, self._config.supported_lanelet2_subtypes
        )

        if self._config.autoware:
            # OpenDRIVE shoulder lanes arrive as LaneletType.BORDER (crdesigner maps both
            # odr 'shoulder' and 'border' to BORDER). Emit them as subtype:road_shoulder
            # (vm-01-15); the CARLA/RoadRunner fixtures contain no genuine 'border' lanes.
            if not subtype_in and LaneletType.BORDER in lanelet.lanelet_type:
                subtype, subtype_in = "road_shoulder", True
            # A regular bicycle lane is tagged subtype:road for Autoware (vm-01-23).
            elif subtype == "bicycle_lane":
                subtype = "road"

        # append left and right way
        self.left_ways[lanelet.lanelet_id] = left_way_id
        self.right_ways[lanelet.lanelet_id] = right_way_id

        # create the way relation
        way_rel = WayRelation(
            self.id_count, left_way_id, right_way_id, tag_dict={"type": "lanelet"}
        )

        # Record OpenDrive -> Lanelet2 ID mapping when the lanelet carries an OpenDrive-
        # derived description (format "road_id.section_id.lane_id.width_id" set by
        # ConversionLaneletNetwork.convert_all_lanelet_ids).
        desc = getattr(lanelet, "description", None)
        if desc is not None:
            parts = str(desc).split(".")
            if len(parts) >= 3:
                try:
                    odr_road_id = int(parts[0])
                    odr_section_id = int(parts[1])
                    odr_lane_id = int(parts[2])
                    self.odr_to_l2_mapping[(odr_road_id, odr_section_id, odr_lane_id)] = way_rel.id_
                except ValueError:
                    pass

        # convert the speed signs
        self._convert_speed_sign(lanelet, way_rel)

        # update the way relation with the subtype, if it exists
        location = None
        if subtype_in is True:
            way_rel.tag_dict.update({"subtype": subtype})

            # set the location tag
            location = "urban" if LaneletType.URBAN in lanelet.lanelet_type else "nonurban"
            way_rel.tag_dict["location"] = location

        if self._config.autoware:
            # Autoware: emit one_way:yes/no + speed_limit on drivable lanes,
            # replacing the granular one_way:<user> tags.
            self._set_autoware_lanelet_tags(lanelet, way_rel, subtype if subtype_in else None, location)
            # Tag junction connector lanelets with their turn direction (vm-03-02).
            turn_direction = self._turn_directions.get(lanelet.lanelet_id)
            if turn_direction is not None:
                way_rel.tag_dict["turn_direction"] = turn_direction
        else:
            # set the overriding tags for bidirectional users
            _set_overriding_tags_for_bidirectional_users(lanelet, way_rel)

        # remember the way-relation id for the intersection_area post-pass (vm-03-01)
        self._lanelet_to_way_rel[lanelet.lanelet_id] = way_rel.id_

        # add the way relation to the osm
        self.osm.add_way_relation(way_rel)

    def _create_nodes(
        self, lanelet: Lanelet, left_way_id: int, right_way_id: int
    ) -> Tuple[List[str], List[str]]:
        """
        Create new nodes for the ways of the lanelet.
        Add them to OSM and return a list of the node ids.
        In case a left or right way already exists, the returned list
        only contains the first and last node of the way.

        :param lanelet: Lanelet of which the right and left vertices should be converted to ways.
        :param left_way_id: Id of a potential shared left way which was already converted.
            if this is not None, the left vertices of the lanelet do not have to be converted again.
        :param right_way_id: Id of a potential right way, similar to left_way_id.
        :return: A tuple of lists of node ids for the left and the right way.
        """
        left_nodes, right_nodes = [], []
        start_index = 0
        end_index = len(lanelet.left_vertices)
        pot_first_left_node, pot_first_right_node = (
            self._get_shared_first_nodes_from_other_lanelets(lanelet)
        )
        pot_last_left_node, pot_last_right_node = self._get_shared_last_nodes_from_other_lanelets(
            lanelet
        )

        if pot_first_left_node:
            start_index = 1
        if pot_last_left_node:
            end_index = -1
        if left_way_id:
            first_left_node: Optional[str]
            last_left_node: Optional[str]
            first_left_node, last_left_node = self._get_first_and_last_nodes_from_way(
                left_way_id, lanelet.adj_left_same_direction
            )
        else:
            first_left_node = pot_first_left_node
            last_left_node = pot_last_left_node
            left_nodes = self._create_nodes_from_vertices(
                lanelet.left_vertices[start_index:end_index]
            )
        if right_way_id:
            first_right_node: Optional[str]
            last_right_node: Optional[str]
            first_right_node, last_right_node = self._get_first_and_last_nodes_from_way(
                right_way_id, lanelet.adj_right_same_direction
            )
        else:
            first_right_node = pot_first_right_node
            last_right_node = pot_last_right_node
            right_nodes = self._create_nodes_from_vertices(
                lanelet.right_vertices[start_index:end_index]
            )

        if first_left_node:
            left_nodes.insert(0, first_left_node)
        if first_right_node:
            right_nodes.insert(0, first_right_node)

        if last_left_node:
            left_nodes.append(last_left_node)
        if last_right_node:
            right_nodes.append(last_right_node)

        return left_nodes, right_nodes

    def _get_first_and_last_nodes_from_way(self, way_id: int, same_dir: bool) -> Tuple[str, str]:
        """
        Get the first and the last node of a way.
        Reverse order of nodes if way is reversed.

        :param way_id: Id of way.
        :param same_dir: True if way is in normal direction, False if it is reversed.
        :return: Tuple with first and last node.
        """
        way = self.osm.find_way_by_id(way_id)
        first_idx, last_idx = (0, -1) if same_dir else (-1, 0)
        return way.nodes[first_idx], way.nodes[last_idx]

    def _create_nodes_from_vertices(self, vertices: List[np.ndarray]) -> List[str]:
        """
        Create nodes and add them to the OSM.

        :param vertices: List of vertices from a lanelet boundary.
        :return: Ids of nodes which were created.
        """
        nodes = []
        for vertex in vertices:
            lat, lon = self.transformer.transform(
                self.origin_utm[0] + vertex[0], self.origin_utm[1] + vertex[1]
            )
            ele = 0  # z-coordinate value
            if (
                len(vertex) > 2
            ):  # if vertex returns z-coordinate (along with x and y), take it into account
                ele = vertex[2]
            if self._config.use_local_coordinates:
                node = Node(
                    self.id_count,
                    lat,
                    lon,
                    ele,
                    autoware=self._config.autoware,
                    local_x=vertex[0],
                    local_y=vertex[1],
                )
            else:
                node = Node(self.id_count, lat, lon, ele, autoware=self._config.autoware)
            nodes.append(node.id_)
            self.osm.add_node(node)
        return nodes

    def _build_turn_directions(self) -> Dict[int, str]:
        """Map each junction connector lanelet id to its turn direction
        ("left"/"straight"/"right") from the CommonRoad intersection model (vm-03-02).

        odr2cr classifies each incoming's successors geometrically when it builds
        the Intersection objects, so the OpenDRIVE <junction> turn semantics are
        recoverable here without re-deriving them. A connector that somehow appears
        in more than one class keeps its first assignment.
        """
        turn_directions = {}
        for intersection in self.lanelet_network.intersections:
            for incoming in intersection.incomings:
                for direction, successors in (
                    ("left", incoming.successors_left),
                    ("straight", incoming.successors_straight),
                    ("right", incoming.successors_right),
                ):
                    for lanelet_id in successors:
                        turn_directions.setdefault(lanelet_id, direction)
        return turn_directions

    @staticmethod
    def _convex_hull(points):
        """Counter-clockwise convex hull (Andrew's monotone chain) of (x, y)
        points; returns the unique points unchanged if fewer than three."""
        pts = sorted({(round(float(p[0]), 6), round(float(p[1]), 6)) for p in points})
        if len(pts) < 3:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    def _add_intersection_areas(self):
        """Synthesize an Autoware intersection_area polygon per OpenDRIVE junction
        (vm-03-01) and reference it from every connector lanelet.

        OpenDRIVE carries no explicit junction outline, so the area is approximated
        by the convex hull of the boundary vertices of the junction's connector
        lanelets (the successor lanelets of each incoming). The hull is emitted as a
        closed ``area:yes`` way tagged ``type:intersection_area``, and each member
        lanelet's relation gets an ``intersection_area`` reference tag.
        """
        for intersection in self.lanelet_network.intersections:
            member_ids = set()
            for incoming in intersection.incomings:
                member_ids |= set(incoming.successors_left)
                member_ids |= set(incoming.successors_straight)
                member_ids |= set(incoming.successors_right)

            points = []
            for lanelet_id in member_ids:
                lanelet = self.lanelet_network.find_lanelet_by_id(lanelet_id)
                if lanelet is None:
                    continue
                points.extend(lanelet.left_vertices)
                points.extend(lanelet.right_vertices)

            hull = self._convex_hull(points)
            if len(hull) < 3:
                continue

            # closed ring: repeat the first vertex as the last node
            ring = [np.array(p) for p in hull] + [np.array(hull[0])]
            nodes = self._create_nodes_from_vertices(ring)
            area_way = Way(
                self.id_count,
                nodes,
                tag_dict={"type": "intersection_area", "area": "yes"},
            )
            self.osm.add_way(area_way)

            # reference the polygon from each connector lanelet
            for lanelet_id in member_ids:
                way_rel_id = self._lanelet_to_way_rel.get(lanelet_id)
                if way_rel_id is None:
                    continue
                way_rel = self.osm.find_way_rel_by_id(way_rel_id)
                if way_rel is not None:
                    way_rel.tag_dict["intersection_area"] = area_way.id_

    @staticmethod
    def _is_crosswalk(lanelet) -> bool:
        """True if a CommonRoad lanelet is a pedestrian crossing."""
        return LaneletType.CROSSWALK in lanelet.lanelet_type

    def _lanelets_crossed_by(self, crosswalk) -> List[int]:
        """Vehicle lanelet ids that a crosswalk lanelet passes over.

        Found by sampling the crosswalk centerline against the lanelet network;
        pedestrian lanelets (other crosswalks, sidewalks/walkways) are excluded
        so the crosswalk regulatory element only binds the crossing road lanelets.
        """
        sample = [np.array(v) for v in crosswalk.center_vertices]
        if not sample:
            return []
        found = set()
        for ids in self.lanelet_network.find_lanelet_by_position(sample):
            found.update(ids)
        crossed = []
        pedestrian = {LaneletType.CROSSWALK, LaneletType.SIDEWALK}
        for lid in found:
            other = self.lanelet_network.find_lanelet_by_id(lid)
            if other is None or pedestrian & other.lanelet_type:
                continue
            crossed.append(lid)
        return crossed

    def _add_crosswalks(self):
        """Emit the Autoware crosswalk profile (vm-05-01) for every crosswalk lanelet.

        A CommonRoad crosswalk lanelet already converts to a ``subtype:crosswalk``
        way-relation, but Autoware additionally needs (a) a ``crosswalk_polygon``
        area describing the painted crossing and (b) a ``subtype:crosswalk``
        regulatory element that links the crossing vehicle lanelets to the
        crosswalk. The polygon is the crosswalk lanelet's own footprint (left
        boundary then right boundary reversed, closed); the crossed vehicle
        lanelets are recovered geometrically from the crosswalk centerline.
        """
        for lanelet in self.lanelet_network.lanelets:
            if not self._is_crosswalk(lanelet):
                continue
            cw_rel_id = self._lanelet_to_way_rel.get(lanelet.lanelet_id)
            if cw_rel_id is None:
                continue

            # crosswalk_polygon footprint: left boundary, right boundary reversed,
            # then closed back to the first vertex.
            ring = list(lanelet.left_vertices) + list(lanelet.right_vertices[::-1])
            ring.append(ring[0])
            nodes = self._create_nodes_from_vertices([np.array(p) for p in ring])
            polygon_way = Way(
                self.id_count,
                nodes,
                tag_dict={"type": "crosswalk_polygon", "area": "yes"},
            )
            self.osm.add_way(polygon_way)

            reg_id = self.id_count
            self.osm.add_regulatory_element(
                RegulatoryElement(
                    reg_id,
                    refers_lanelets=[cw_rel_id],
                    crosswalk_polygon=[polygon_way.id_],
                    tag_dict={"type": "regulatory_element", "subtype": "crosswalk"},
                )
            )

            # bind the crosswalk reg-elem to every vehicle lanelet it crosses
            for road_id in self._lanelets_crossed_by(lanelet):
                way_rel_id = self._lanelet_to_way_rel.get(road_id)
                if way_rel_id is None:
                    continue
                way_rel = self.osm.find_way_rel_by_id(way_rel_id)
                if way_rel is None or reg_id in way_rel.regulatory_elements:
                    continue
                way_rel.regulatory_elements.append(reg_id)

    def _shared_way_with_neighbours(self, lanelet, my_vertices, my_line_marking, my_side):
        """Return an already-created neighbour boundary ``way`` geometrically
        coincident with this lanelet's ``my_side`` ("left"/"right") boundary, or None.

        Sharing is matched by **geometry**, not by adj_left/adj_right orientation:
        in left-hand-traffic maps the CommonRoad left/right vertex labels are
        inverted relative to the driving direction, so the old label-based pairing
        compared the wrong boundaries and deduplicated nothing. Here the boundary
        is tested against each neighbour's left and right boundary in both
        orientations; the first coincident way is reused and tagged with the
        combined line marking.
        """
        tolerance = self._config.ways_are_equal_tolerance
        neighbour_ids = [adj for adj in (lanelet.adj_left, lanelet.adj_right) if adj is not None]
        for adj_id in neighbour_ids:
            adj = self.lanelet_network.find_lanelet_by_id(adj_id)
            for adj_side in ("left", "right"):
                way_id = (self.left_ways if adj_side == "left" else self.right_ways).get(adj_id)
                if not way_id:
                    continue
                adj_vertices = adj.left_vertices if adj_side == "left" else adj.right_vertices
                if _vertices_are_equal(my_vertices, adj_vertices, tolerance) or _vertices_are_equal(
                    my_vertices, adj_vertices[::-1], tolerance
                ):
                    adj_line_marking = (
                        adj.line_marking_left_vertices
                        if adj_side == "left"
                        else adj.line_marking_right_vertices
                    )
                    self._set_shared_way_tags(way_id, my_line_marking, adj_line_marking, my_side)
                    return way_id

        # Geometric fallback: CommonRoad does not always set opposite-direction
        # adjacency (odr2cr leaves ~1/3 of coincident opposing pairs unlinked, e.g.
        # across roads / junction connectors), so the loop above misses them and the
        # opposing centerline would be emitted as two distinct coincident ways
        # (vm-01-04). Search the endpoint-bucket index of already-created boundaries
        # for one coincident with this boundary (either orientation) and reuse it.
        # The match still requires full equality within ways_are_equal_tolerance
        # (~1 mm), so only genuinely identical polylines merge.
        for cand_id, cand_vertices, cand_marking in self._boundary_index.get(
            self._endpoint_key(my_vertices), []
        ):
            if _vertices_are_equal(my_vertices, cand_vertices, tolerance) or _vertices_are_equal(
                my_vertices, cand_vertices[::-1], tolerance
            ):
                self._set_shared_way_tags(cand_id, my_line_marking, cand_marking, my_side)
                return cand_id
        return None

    @staticmethod
    def _endpoint_key(vertices):
        """Bucket key from a polyline's two endpoints, rounded to a 1 m grid and
        sorted so a polyline and its reverse land in the same bucket."""
        first, last = vertices[0], vertices[-1]
        p0 = (round(float(first[0])), round(float(first[1])))
        p1 = (round(float(last[0])), round(float(last[1])))
        return tuple(sorted((p0, p1)))

    def _index_boundary(self, way_id, vertices, line_marking):
        """Register a freshly created boundary way for the geometric fallback."""
        self._boundary_index[self._endpoint_key(vertices)].append(
            (way_id, vertices, line_marking)
        )

    def _set_shared_way_tags(self, way_id, my_line_marking, adj_line_marking, my_side):
        """Tag a shared boundary ``way`` with the L2 type/subtype derived from the
        two lanelets' line markings (combined when both are known)."""
        type_lanelet, subtype_lanelet = _line_marking_to_type_subtype_vertices(my_line_marking)
        type_adj, subtype_adj = _line_marking_to_type_subtype_vertices(adj_line_marking)
        if type_lanelet != "unknown":
            if type_adj != "unknown":
                # combine the two halves of the shared marking; keep the existing
                # own-then-neighbour (right) / neighbour-then-own (left) ordering so
                # directional subtypes (solid_dashed vs dashed_solid) stay consistent.
                subtype = (
                    _combine_subtypes(subtype_lanelet, subtype_adj)
                    if my_side == "right"
                    else _combine_subtypes(subtype_adj, subtype_lanelet)
                )
            else:
                subtype = subtype_lanelet
            self.osm.ways[way_id].tag_dict = {"type": type_lanelet, "subtype": subtype}
        elif type_adj != "unknown":
            self.osm.ways[way_id].tag_dict = {"type": type_adj, "subtype": subtype_adj}

    def _get_potential_right_way(self, lanelet) -> Union[None, int]:
        """
        Check if a shared right boundary with another lanelet can be transformed
        to the same way.

        :param lanelet: Lanelet of which right boundary should be converted to a way.
        :return: Id of a way which can be shared, else None if it is not possible.
        """
        return self._shared_way_with_neighbours(
            lanelet, lanelet.right_vertices, lanelet.line_marking_right_vertices, "right"
        )

    def _get_potential_left_way(self, lanelet) -> Union[None, int]:
        """
        Check if a shared left boundary with another lanelet can be transformed
        to the same way.

        :param lanelet: Lanelet of which left boundary should be converted to a way.
        :return: Id of a way which can be shared, else None if it is not possible.
        """
        return self._shared_way_with_neighbours(
            lanelet, lanelet.left_vertices, lanelet.line_marking_left_vertices, "left"
        )

    def _get_shared_first_nodes_from_other_lanelets(
        self, lanelet: Lanelet
    ) -> Tuple[Union[str, None], Union[str, None]]:
        """
        Get already created nodes from other lanelets which could also
        be used by this lanelet as first nodes.

        :param lanelet: Lanelet for which shared nodes should be found.
        :return: Id of first left and first right node if they exist.
        """
        if lanelet.predecessor:
            for lanelet_id in lanelet.predecessor:
                first_left_node, first_right_node = self.last_nodes.get(lanelet_id, (None, None))
                if first_left_node:
                    return first_left_node, first_right_node
            for pred_id in lanelet.predecessor:
                pred = self.lanelet_network.find_lanelet_by_id(pred_id)
                for succ_id in pred.successor:
                    first_left_node, first_right_node = self.first_nodes.get(succ_id, (None, None))
                    if first_left_node:
                        return first_left_node, first_right_node
        return None, None

    def _get_shared_last_nodes_from_other_lanelets(
        self, lanelet: Lanelet
    ) -> Tuple[Union[str, None], Union[str, None]]:
        """
        Get already created nodes from other lanelets which could also
        be used by this lanelet as last nodes.

        :param lanelet: Lanelet for which shared nodes should be found.
        :return: Id of last left and last right node if they exist.
        """
        if lanelet.successor:
            for lanelet_id in lanelet.successor:
                last_left_node, last_right_node = self.first_nodes.get(lanelet_id, (None, None))
                if last_left_node:
                    return last_left_node, last_right_node
            for succ_id in lanelet.successor:
                succ = self.lanelet_network.find_lanelet_by_id(succ_id)
                for pred_id in succ.predecessor:
                    last_left_node, last_right_node = self.last_nodes.get(pred_id, (None, None))
                    if last_left_node:
                        return last_left_node, last_right_node

        return None, None

    def _convert_speed_sign(self, lanelet: Lanelet, way_rel: WayRelation):
        """
        Convert speed signs(CR) into regulatory elements(L2)

        :param lanelet: Lanelet(CR) that contains the traffic sign
        :param way_rel: Way relation, in which regulatory elements will append newly converted speed sign.
        """
        speed_sign_ids = []
        # check if the lanelet has a traffic sign (speed limit) assigned to it.
        # if so, create a regulatory element for speed limit and reference it
        for traffic_sign in lanelet.traffic_signs:
            for sign in self.lanelet_network.traffic_signs:
                if sign.traffic_sign_id == traffic_sign:
                    if (
                        list(sign.traffic_sign_elements)[0].traffic_sign_element_id.name
                        == "MAX_SPEED"
                    ):
                        # found a max speed sign, create a reg. element
                        speed_sign_id = self.id_count
                        speed_sign_ids.append(speed_sign_id)
                        max_speed = sign.traffic_sign_elements[0].additional_values[0]
                        self.osm.add_regulatory_element(
                            RegulatoryElement(
                                speed_sign_id,
                                tag_dict={
                                    "sign_type": max_speed,
                                    "subtype": "speed_limit",
                                    "type": "regulatory_element",
                                },
                            )
                        )
        for speed_sign_id in speed_sign_ids:
            way_rel.regulatory_elements.append(str(speed_sign_id))

    def _lanelet_max_speed_kmh(self, lanelet: Lanelet) -> Optional[float]:
        """Return the lanelet's MAX_SPEED sign value in km/h, or None if it has none."""
        for traffic_sign_id in lanelet.traffic_signs:
            for sign in self.lanelet_network.traffic_signs:
                if sign.traffic_sign_id != traffic_sign_id:
                    continue
                element = list(sign.traffic_sign_elements)[0]
                if element.traffic_sign_element_id.name == "MAX_SPEED":
                    try:
                        # CommonRoad stores MAX_SPEED in m/s.
                        return float(element.additional_values[0]) * _MS_TO_KMH
                    except (ValueError, IndexError):
                        return None
        return None

    def _set_autoware_lanelet_tags(
        self, lanelet: Lanelet, way_rel: WayRelation, subtype: Optional[str], location: Optional[str]
    ):
        """
        Add Autoware-required lanelet tags (one_way, speed_limit) to a drivable lane.

        - ``one_way``: ``yes`` unless the lane carries bidirectional users. This replaces
          the granular ``one_way:<user>`` tags that the non-Autoware path emits.
        - ``speed_limit`` (km/h): taken from a MAX_SPEED sign on the lanelet if present,
          otherwise a per-location default (vm-01-01 / vm-01-09).
        """
        if subtype not in AUTOWARE_DRIVING_SUBTYPES:
            return

        way_rel.tag_dict["one_way"] = "no" if lanelet.user_bidirectional else "yes"

        speed_kmh = self._lanelet_max_speed_kmh(lanelet)
        if speed_kmh is None:
            defaults = getattr(
                self._config, "autoware_default_speed_kmh", AUTOWARE_DEFAULT_SPEED_KMH
            )
            speed_kmh = defaults.get(location, defaults.get("urban", AUTOWARE_DEFAULT_SPEED_KMH["urban"]))
        # Whole km/h reads cleanly in Autoware; sign values derived from mph are not round.
        way_rel.tag_dict["speed_limit"] = f"{speed_kmh:.0f}"

    def _append_lane_change_tags(self):
        """
        Function that appends the lane change tags to osm ways based on the linemarking of the way.
        Possibility of a lane change has been copied from the Lanelet2 documentation.
        The lane change tags are being used by Autoware.
        """
        # Default applied where the marking does not determine lane_change; user-configurable.
        default_lane_change = getattr(self._config, "autoware_default_lane_change", "no")
        # Point/area features are not lane boundaries — lane_change is meaningless on them.
        non_boundary = {"traffic_light", "traffic_sign", "light_bulbs", "stop_line",
                        "intersection_area"}
        ways = list(self.osm.ways.values())
        for way in ways:
            if way.tag_dict.get("type") in non_boundary:
                continue
            if way.tag_dict:
                if "subtype" in way.tag_dict:
                    subtype = way.tag_dict["subtype"]
                    if subtype == "dashed" or subtype == "dashed_dashed":
                        way.tag_dict["lane_change"] = "yes"
                    elif subtype == "dashed_solid":
                        way.tag_dict["lane_change"] = "left->right: yes"
                    elif subtype == "solid_dashed":
                        way.tag_dict["lane_change"] = "right->left: yes"
                    elif (
                        way.tag_dict["type"] != "traffic_light"
                        and way.tag_dict["type"] != "traffic_sign"
                    ):
                        # a non-dashed lane marking (solid, road_border, …) forbids a lane change
                        way.tag_dict["lane_change"] = "no"
                    else:
                        # regulatory-element way (traffic light/sign), not a lane boundary
                        way.tag_dict["lane_change"] = "no"
                else:
                    # no line-marking subtype to derive from -> configurable default
                    way.tag_dict["lane_change"] = default_lane_change
            else:
                # no line marking at all -> configurable default
                way.tag_dict["lane_change"] = default_lane_change
