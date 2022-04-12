import json

import pandas as pd
import pickle
import torch
from shapely.geometry import Polygon

from graph import Graph

""" 
Code to read and search the data files  
"""


def read_custom_triples(custom_triples_path):
    custom_triples = pd.read_csv(custom_triples_path, sep='|')
    custom_triples.columns = ['head', 'rel', 'tail']
    return custom_triples


def find_class(classes_path, entity):
    with open(classes_path) as json_file:
        data = json.load(json_file)
        return data[entity]


def find_id(file_path, row):
    data = pd.read_csv(file_path, sep=" ", header=None)
    data.columns = ['data', 'id']
    return data[data.data == row].id


def find_node_maps(classes_path, entitiesID_path):
    entity_ids = pd.read_csv(entitiesID_path, sep=" ", header=None)
    entity_ids.columns = ['entity', 'id']

    with open(classes_path) as json_file:
        types = json.load(json_file)

        # Create {type : {entity_id : local_entity_id}}
        node_maps = dict()
        for entity in types:

            entity_type = types[entity]
            if entity_type not in node_maps:
                node_maps[entity_type] = dict()

            entity_id = entity_ids[entity_ids.entity == entity].id.values[0]
            if entity_id not in node_maps[entity_type]:
                node_maps[entity_type][entity_id] = -1

        # Set local entity ids for each type
        for e_type in node_maps:
            for local_entity_id, entity_id in enumerate(node_maps[e_type]):
                node_maps[e_type][entity_id] = local_entity_id

        return node_maps


def read_id2geo(in_path):
    """
    Dictionary in json contains key: geo instance id | value: dictionary with keys:
    area
    length_parallel
    length_orthogonal
    rectangle_center
    unit_vector
    unit_vector_angle
    corner_points

    param in_path: json filepath
    :return: dictionary of key: id | values: [center] of box
    """

    # Load geo locations
    with open(in_path) as file:
        data = json.load(file)

    id2geo = dict()

    # For each geo location
    for geo_id in data.keys():
        center = data[geo_id]['rectangle_center']
        # poly = Polygon(corners)
        # random_point = random_points_within(poly, 1)

        id2geo[geo_id] = center

    return id2geo


def read_id2extent(in_path):
    """
        Dictionary in json contains key: geo instance id | value: dictionary with keys:
        area
        length_parallel
        length_orthogonal
        rectangle_center
        unit_vector
        unit_vector_angle
        corner_points
        :param in_path: json filepath
        :return: dictionary of key: id | values: [northeast, southwest] box coordinates
        """

    # Load geo locations
    with open(in_path) as file:
        data = json.load(file)

    id2extent = dict()

    # For each geo location
    for geo_id in data.keys():
        corners = data[geo_id]['corner_points']

        # Find the northeast and the southwest corners
        northeast = corners[2]
        southwest = corners[0]

        id2extent[geo_id] = [northeast, southwest]

    return id2extent


def read_graph(graph_path):
    [feature_dims, relations, adj_lists, feature_modules, node_maps, inv_rel] = pickle_load(graph_path)

    def features(nodes, e_type):
        return feature_modules[e_type](
            torch.autograd.Variable(torch.LongTensor([node_maps[e_type][n] for n in nodes]).to('cuda') + 1))

    graph = Graph(features, feature_dims, relations, adj_lists, inv_rel)
    return graph, feature_modules, node_maps


def pickle_load(pickle_filepath):
    with open(pickle_filepath, "rb") as f:
        obj = pickle.load(f)
    return obj


def pickle_dump(obj, pickle_filepath):
    with open(pickle_filepath, "wb") as f:
        pickle.dump(obj, f, protocol=2)
