import json
import re

import pandas as pd
import os

import torch
from rdflib import Graph, Literal, RDF, URIRef
from Yago2GeoDatasetHelpers.dataset_helper_functions import pickle_dump, read_custom_triples, find_node_maps
from Yago2GeoDatasetHelpers.MinimumBoundingBox import minimum_bounding_box

""" 
Create files: 
    united_triples.nt (all the triples incl materialization, materialization geometries changed to corresponding entities) 
    geo_classes.ttl (all the entities with their properties incl geometries)
    entity2id.txt (entities from geo_classes.ttl with their ids)
    relation2id.txt (unique relations from united_triples.nt with their ids)
    relation2inverse.json (relations with their inverses '_inverse>')
    rid2inverse.json (relation id with the ids of their inverses)
    id2geo.json (entity ids with their geometries)
    entity2type.json (entities with their types)
    custom_triples.txt (all the triples in the form of '(head_id, (head_type, relation_id, tail_type), tail_id)')
    graph.pkl (the graph object containing all the KG)
"""


class MyDict(dict):
    def __missing__(self, key):
        return key


def relation_inverse(relation):
    return relation.replace('>', '_inverse>') if '_inverse>' not in relation else relation.replace('_inverse>', '>')


def transform(data):
    data = data.replace('kr.di.uoa.gr', 'kr_di_uoa_gr', regex=True)
    data = data.replace('www.opengis.net', 'www_opengis_net', regex=True)
    data = data.replace('knowledge.org', 'knowledge_org', regex=True)
    data = data.replace('St\.', 'St', regex=True)
    return data


def read_triples(triples_path, change_data=True):
    print(f'reading {triples_path}')
    data = pd.read_csv(triples_path, sep=" ", header=0)
    data.columns = ['head', 'relation', 'tail', '.']
    data.drop(columns=['.'])
    if change_data:
        data = transform(data)
    return data


def read_geo_classes(geo_classes_path, change_data=True):
    geometry2polygon = {}
    entity2type = {}
    entities_dict = {}
    doubles_dict = {}
    entities_without_types = []

    for file in os.listdir(geo_classes_path):
        g = Graph()
        print(f'reading {file}')
        g.parse(geo_classes_path+file, format='ttl')

        for geometry, polygon in g.subject_objects(predicate=URIRef('http://www.opengis.net/ont/geosparql#asWKT')):
            geometry2polygon[geometry] = polygon

        for entity, type in g.subject_objects(predicate=URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')):
            if entity in entity2type:  # If we have this entity from another dataset keep the old type
                continue
            entity2type[entity] = type

        for entity, geometry in g.subject_objects(predicate=URIRef('http://www.opengis.net/ont/geosparql#hasGeometry')):
            if entity in entities_dict:
                doubles_dict[entity] = [entity, geometry]
                continue
            if entity not in entity2type.keys():
                entities_without_types.append(entity)
                continue
            entities_dict[entity] = [entity, entity2type[entity], geometry, geometry2polygon[geometry]]

    print(f'These entities have no type: \n\t{entities_without_types}')
    entities_double = pd.DataFrame.from_dict(doubles_dict, orient='index', columns=['entity', 'geometry']).reset_index(drop=True).reset_index().rename(columns={'index': 'id'})

    entities = pd.DataFrame.from_dict(entities_dict, orient='index', columns=['entity', 'type', 'geometry', 'polygon']).reset_index(drop=True).reset_index().rename(columns={'index': 'id'})

    if change_data:
        entities = transform(entities)
        entities_double = transform(entities_double)

    return entities, entities_double


def unite_files(triples_path, geo_classes_path, out_path):
    data = pd.DataFrame()
    # Join all the triples in one file
    for root, directories, files in os.walk(triples_path):
        for name in files:
            data = data.append(read_triples(os.path.join(root, name)))

    data.to_csv(out_path + 'united_triples.nt', sep=' ', index=False)

    # Join all the attributes and geometries of the entities in one file / OUT
    # geo_classes_file = open(out_path + 'geo_classes.ttl', 'a')
    # for file in os.listdir(geo_classes_path):
    #     with open(geo_classes_path + file) as f:
    #         data = f.read()
    #         geo_classes_file.write(data)

    return out_path + 'united_triples.nt'  # , out_path + 'geo_classes.ttl'


def make_id_files(triples_path, geo_classes_path, out_path):
    entities_attributes, entities_attributes_doubles = read_geo_classes(geo_classes_path)
    entities_attributes[['entity', 'type', 'geometry']] = entities_attributes[['entity', 'type', 'geometry']].apply(lambda x: '<'+x+'>')
    entities_attributes_doubles[['entity', 'geometry']] = entities_attributes_doubles[['entity', 'geometry']].apply(lambda x: '<'+x+'>')
    triples = read_triples(triples_path)

    # Create entity to id file
    print('Create entity to id file..')
    entities = entities_attributes[['entity', 'id']]
    entities['id'] = entities['id'].map(str)
    entities.to_csv(out_path + 'entity2id.txt', index=False, sep=' ')

    # Create relation and inverses to id
    print('Create relation and inverses to id..')
    relations = triples['relation'].drop_duplicates().reset_index(drop=True).reset_index().rename(
        columns={'index': 'id'})
    relations = relations[['relation', 'id']]
    relations['id'] = relations['id'].map(str)

    rel = list(relations['relation'])
    for r in rel:
        relations = relations.append({'relation': relation_inverse(r), 'id': str(len(relations))}, ignore_index=True)
    relations.to_csv(out_path + 'relation2id.txt', index=False, sep=' ')

    # Create relation to inverse
    print('Create relation to inverse..')
    relation2inverse = {}
    for relation in relations['relation']:
        relation2inverse[relation] = relation_inverse(relation)

    with open(out_path + 'relation2inverse.json', 'w') as file:
        json.dump(relation2inverse, file)

    # Create relation id to inverse id
    print('Create relation id to inverse id..')
    relation_id2inverse_id = {}
    for i, [relation, id] in relations.iterrows():
        relation_id2inverse_id[str(id)] = str(relations[relations['relation'] == relation_inverse(relation)]['id'].item())

    with open(out_path + 'rid2inverse.json', 'w') as file:
        json.dump(relation_id2inverse_id, file)

    # Change geometries in triples to entities
    print('Change geometries in triples to entities..')
    geometry2entity = MyDict(zip(entities_attributes['geometry'].append(entities_attributes_doubles['geometry']), entities_attributes['entity'].append(entities_attributes_doubles['entity'])))
    triples['head'] = triples['head'].map(geometry2entity)
    triples['tail'] = triples['tail'].map(geometry2entity)
    triples.to_csv(triples_path, sep=' ', index=False)

    # Create id to geometries
    print('Create id to geometries..')
    id2polygon = dict(zip(entities_attributes['id'], entities_attributes['polygon']))
    count = 0
    id2geometry = dict()
    for id in id2polygon.keys():
        # print(f'Minimum bounding box for Geometry with id {id}')
        # polygon = [coord_pair.lstrip().split(' ') for coord_pair in
        #            id2geometry[id].split('((')[1].split('))')[0].replace('(', '').replace(')', '').split(
        #                ',')]
        polygon = [[float(n) for n in s.split(' ')] for s in re.findall('\-?\d*\.?\d+\s\-?\d*\.?\d+', id2polygon[id])]  # Take only the pairs of coordinates
        # polygon = [[float(coord) for coord in pair] for pair in polygon]  # Convert strings to floats
        if len(polygon) <= 2:
            count += 1
            print(f'Polygons smaller than 2 points : {count}')
            continue
        id2geometry[str(id)] = minimum_bounding_box(polygon)

    with open(out_path + 'id2geo.json', 'w') as file:
        json.dump(id2geometry, file)

    # Create entity to type
    print('Create entity to type..')
    with open(out_path + 'entity2type.json', 'w') as file:
        json.dump(dict(zip(entities_attributes['entity'], entities_attributes['type'])), file)

    # Create id to type
    print('Create id to type..')
    with open(out_path + 'id2type.json', 'w') as file:
        json.dump(dict(zip(entities_attributes['id'], entities_attributes['type'])), file)

    return


def make_custom_triples(triples_path, classes_path, relationsID_path, entitiesID_path, out_path):
    # Read the triples
    data = read_triples(triples_path)

    # Read the relations to id's
    relations = pd.read_csv(relationsID_path, sep=" ", header=None)
    relations.columns = ['relation', 'id']

    # Read the entities to id's
    entities = pd.read_csv(entitiesID_path, sep=" ", header=None)
    entities.columns = ['entity', 'id']

    # Open the classes json
    with open(classes_path) as json_file:
        classes = json.load(json_file)

        # for each relation find its head and tail classes and head relation tail id's
        custom_triples = list()
        for index, row in data.iterrows():
            if index < 0:
                continue
            elif index % 10000 == 0:
                df = pd.DataFrame(custom_triples)
                df.to_csv(out_path + 'custom_triples.txt', index=False, sep='|', mode='a', header=False)
                custom_triples = list()
                print(f'Saved {index-1} triples')
            head_class = classes[row['head']]
            tail_class = classes[row['tail']]
            relation_id = str(relations[relations.relation == row['relation']].id.values[0])
            head_id = str(entities[entities.entity == row['head']].id.values[0])
            tail_id = str(entities[entities.entity == row['tail']].id.values[0])

            # (head_id, (head_class, relation_id, tail_class), tail_id)
            custom_triple = (head_id, (head_class, relation_id, tail_class), tail_id)

            custom_triples.append(custom_triple)

    df = pd.DataFrame(custom_triples)
    df.columns = ['head_id', 'triple', 'tail_id']
    df['head_id'] = df['head_id'].map(str)
    df['tail_id'] = df['tail_id'].map(str)
    df.to_csv(out_path + 'custom_triples.txt', index=False, sep='|', mode='a', header=False)

    return out_path + 'custom_triples.txt'


def custom_triples_split(triples_path, out_path):
    # Read all triples and put them in format [(head_id, (head_type, rel_id, tail_type), tail_id)]
    all_triples = pd.read_csv(triples_path, sep='|', header=None)

    all_triples.columns = ['head_id', 'triple', 'tail_id']
    all_triples = [(str(row['head_id']), tuple([x.strip(" (')") for x in row['triple'].split(',')]), str(row['tail_id'])) for
                   index, row in all_triples.iterrows()]

    # Split triples to train / valid / test
    train_triples = all_triples[:int(0.7 * len(all_triples))]
    val_triples = all_triples[int(0.7 * len(all_triples)) + 1:int(0.8 * len(all_triples))]
    test_triples = all_triples[int(0.8 * len(all_triples)) + 1:len(all_triples)]

    # Write them in pkl files

    pickle_dump(train_triples, out_path + 'train_triples.pkl')
    pickle_dump(val_triples, out_path + 'valid_triples.pkl')
    pickle_dump(test_triples, out_path + 'test_triples.pkl')

    return


def make_graph(custom_triples_path, classes_path, entitiesID_path, graph_path, rid2inverse_path, id2type_path,
               embed_dim=128):
    custom_triples = read_custom_triples(custom_triples_path)
    custom_triples = custom_triples[custom_triples['rel'] != 'triple']

    print('Creating Graph..')

    with open(rid2inverse_path) as json_file:
        inv_rel_ids = json.load(json_file)

    with open(id2type_path) as json_file:
        id2type = json.load(json_file)

    adj_lists = dict()
    relations = dict()
    for index, custom_triple in custom_triples.iterrows():
        # Separate head, relation and tail
        rel = custom_triple[1].strip("(')").replace("'", "").replace(' ', '').split(',')  # (head_class, pred_id, tail_class)
        rel_inv = (rel[2], inv_rel_ids[rel[1]], rel[0])

        rel = tuple(rel)

        head = str(custom_triple['head_id'])  # head_id
        tail = str(custom_triple['tail_id'])  # tail.id

        # Create Adjacent Lists {(head_class, pred_id, Tail_class) : {head_id : [tail entity id's]}} and inverse
        if rel not in adj_lists:
            adj_lists[rel] = dict()

        if rel_inv not in adj_lists:
            adj_lists[rel_inv] = dict()

        if head not in adj_lists[rel]:
            adj_lists[rel][head] = list()

        if tail not in adj_lists[rel_inv]:
            adj_lists[rel_inv][tail] = list()

        adj_lists[rel][head].append(tail)
        adj_lists[rel_inv][tail].append(head)

        # Create Relations {head_class : [[tail_class, pred_id]]} and inverse
        if rel[0] not in relations:
            relations[rel[0]] = list()

        if rel_inv[0] not in relations:
            relations[rel_inv[0]] = list()

        relations[rel[0]].append((rel[2], rel[1]))
        relations[rel_inv[0]].append((rel_inv[2], rel_inv[1]))

    # Create Node Maps {class : {entity_id : local_entity_id}}
    node_maps = find_node_maps(classes_path, entitiesID_path)

    # Delete duplicates from relation lists
    for key in relations:
        relations[key] = set(relations[key])

    # ???
    for m in node_maps:
        node_maps[m][-1] = -1

    # For each type set feature dimension equal to embed_dim
    feature_dims = {m: embed_dim for m in relations}

    # # For each type
    # feature_modules = dict()
    # for e_type in relations:
    #     # initialize embedding matrix for each type with (num of embeddings = num of ent per type + 1, embed_dim = 10)
    #     feature_modules[e_type] = torch.nn.Embedding(len(node_maps[e_type]) + 1, embed_dim)
    #
    #     # define embedding initialization method: normal dist
    #     feature_modules[e_type].weight.data.normal_(0, 1. / embed_dim)

    pickle_dump([feature_dims, relations, adj_lists, node_maps, inv_rel_ids, id2type], graph_path)

    return


def create_id2type(entitiesID_path, classes_path, id2type_path):
    id2type = dict()

    # Create id to type
    entities_id = pd.read_csv(entitiesID_path, sep="\t", header=None)
    entities_id.columns = ['entity', 'id']

    with open(classes_path) as json_file:
        classes = json.load(json_file)

        for index, row in entities_id.iterrows():
            id2type[row['id']] = classes[row['entity']]

    with open(id2type_path, 'w') as file:
        json.dump(id2type, file)
