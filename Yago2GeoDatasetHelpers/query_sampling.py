import pickle
from collections import defaultdict
from multiprocessing import Process
from os import path

import json

from Yago2GeoDatasetHelpers.dataset_helper_functions import read_graph, pickle_load
from graph import Query

""" Code to sample queries from the dataset for training and testing """


def make_single_edge_query(graph, edge, neg_sample_size):
    neg_samples = graph.get_negative_edge_samples(edge, neg_sample_size)
    return Query(("1-chain", edge), neg_samples, None, neg_sample_size, keep_graph=True)


def make_single_edge_query_data(data_path, graph_path, neg_sample_size):
    """
    1. Load graph-data.pkl for the same format
    2. Load training/valid/testing triples, a list of edge (head id, (domain type, predicate, range type), tail id)
    """
    print("Loading graph...")
    graph, _, _ = read_graph(graph_path)

    print("Load training/valid/testing triples...")
    train_triples = pickle_load(data_path + "train_triples.pkl")
    valid_triples = pickle_load(data_path + "valid_triples.pkl")
    test_triples = pickle_load(data_path + "test_triples.pkl")

    print("Getting full negative samples (for APR evaluation) and make queries...")
    valid_queries = [make_single_edge_query(graph, edge, neg_sample_size) for edge in valid_triples]
    test_queries = [make_single_edge_query(graph, edge, neg_sample_size) for edge in test_triples]

    print("Getting one negative samples (for AUC evaluation) and make queries...")
    valid_queries += [make_single_edge_query(graph, edge, 1) for edge in valid_triples]
    test_queries += [make_single_edge_query(graph, edge, 1) for edge in test_triples]

    print("Dumping valid/test 1-chain queries")
    pickle.dump([q.serialize() for q in valid_queries], open(data_path + "val_edges.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries], open(data_path + "test_edges.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

    print("Dumping train 1-chain queries")
    train_queries = [Query(("1-chain", e), None, None, keep_graph=True) for e in train_triples]
    pickle.dump([q.serialize() for q in train_queries], open(data_path + "train_edges.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)

    print("Finish making training/valid/testing 1-chain queries")


def make_multiedge_query_data(data_path, graph_path, num_workers, samples_per_worker, mp_result_dir=None, id2geo=None):
    """
    Args:
        id2geo: node id => [longitude, latitude]
                if not None, we sample geographic query with target node as geographic entity
    """
    graph, _, _ = read_graph(graph_path)
    queries_2, queries_3 = parallel_sample(graph,
                                           num_workers,
                                           samples_per_worker,
                                           data_path,
                                           test=False,
                                           mp_result_dir=mp_result_dir,
                                           id2geo=id2geo)

    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""
    pickle.dump([q.serialize() for q in queries_2],
                open(data_path + "train_queries_2{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3],
                open(data_path + "train_queries_3{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def make_inter_query_data(data_path, graph_path, num_workers, samples_per_worker, max_inter_size=5, mp_result_dir=None,
                          id2geo=None):
    """
    This just like sample x-inter query from KG, Learning projection and intersection operator from the KG directly
    Args:
        id2geo: node id => [longitude, latitude]
                if not None, we sample geographic query with target node as geographic entity
    """
    graph, _, _ = read_graph(graph_path)
    queries_dict = parallel_inter_query_sample(graph, num_workers, samples_per_worker, data_path,
                                               max_inter_size=max_inter_size,
                                               test=False,
                                               mp_result_dir=mp_result_dir,
                                               id2geo=id2geo)

    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""
    for arity in queries_dict:
        pickle.dump([q.serialize() for q in queries_dict[arity]],
                    open(data_path + "/train_inter_queries_{:d}{:s}.pkl".format(arity, file_postfix), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)


def sample_new_clean(data_path, graph_path, id2geo=None):
    graph_loader = lambda: read_graph(graph_path)[0]
    sample_clean_test(graph_loader, data_path, id2geo=id2geo)


def load_queries(data_file, keep_graph=False):
    """
    1. read query method
    Read query file as a list of Query object
    """
    raw_info = pickle.load(open(data_file, "rb"))
    return [Query.deserialize(info, keep_graph=keep_graph) for info in raw_info]


def load_queries_by_formula(data_file, keep_graph=True):
    """
    2. read query method
    Read query file as a dict
    key: query type
    value: a dict()
        key: formula template
        value: the query object
    """
    if path.exists(data_file):
        raw_info = pickle.load(open(data_file, "rb"))
        queries = defaultdict(lambda: defaultdict(list))
        i = 0
        for raw_query in raw_info:
            i += 1
            query = Query.deserialize(raw_query, keep_graph=keep_graph)
            queries[query.formula.query_type][query.formula].append(query)
            # if i > 4:
            #     print('Keeping 5')
            #     break
        return queries
    else:
        return None


def load_queries_by_type(data_file, keep_graph=True):
    """
    3. read query method
    Read query file as a dict
    key: query type
    value: a list of Query object
    """
    raw_info = pickle.load(open(data_file, "rb"))
    queries = defaultdict(list)
    for raw_query in raw_info:
        query = Query.deserialize(raw_query, keep_graph=keep_graph)
        queries[query.formula.query_type].append(query)
        # print('Keeping 1')
        # break
    return queries


def load_test_queries_by_formula(data_file, keep_graph=False):
    """
    4. read query method
    Read query file as a dict
    key: "full_neg" (full negative sample) or "one_neg" (only one negative sample)
    value: a dict()
        key: query type
        value: a dict()
            key: formula template
            value: the query object
    """
    if path.exists(data_file):
        raw_info = pickle.load(open(data_file, "rb"))
        queries = {"full_neg": defaultdict(lambda: defaultdict(list)),
                   "one_neg": defaultdict(lambda: defaultdict(list))}
        for raw_query in raw_info:
            # neg_type = "full_neg" if len(raw_query[1]) > 1 else "one_neg"
            if len(raw_query[1]) > 1:
                neg_type = "full_neg"
            elif len(raw_query[1]) == 1:
                neg_type = "one_neg"
            else:
                continue
            query = Query.deserialize(raw_query, keep_graph=keep_graph)
            queries[neg_type][query.formula.query_type][query.formula].append(query)
            # print('Keeping 1')
            # break
        return queries
    else:
        return None


def json_load(filepath):
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    return data


def json_dump(data, filepath, pretty_format=True):
    with open(filepath, 'w') as fw:
        if pretty_format:
            json.dump(data, fw, indent=2, sort_keys=True)
        else:
            json.dump(data, fw)


def sample_clean_test(graph_loader, data_dir, num_test_query=10000, num_val_query=1000, id2geo=None):
    """
    Given graph_data.pkl, testing and validation edge data, sampling 2 and 3 edges testing/validation queries and save them on disk
    Args:
        graph_loader: a function which load the graph data, graph_data.pkl
        data_dir: the direction which to read testing and validation edge data, and dump the sampled query data
        num_test_query: the total number of test query to be generated
        num_val_query: the total number of validation query to be generated
        id2geo: node id => [longitude, latitude], If not None, generate geographic queries with target node has coordinates
    """
    # load the graph data into training and testing graph
    print("load the graph data into training and testing graph")
    train_graph = graph_loader()
    test_graph = graph_loader()
    # load the validation and testing edges which need to be deleted from training graph
    print("load the validation and testing edges which need to be deleted from training graph")
    test_edges = load_queries(data_dir + "/test_edges.pkl")
    val_edges = load_queries(data_dir + "/val_edges.pkl")
    # remove all testing and validation edges from the training graph
    print("remove test/valid from train graph")
    train_graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges + val_edges])
    # Sampling 2 edges testing and validation queries
    print("Sampling 2 edges testing queries")
    test_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], num_test_query * 9 / 10, 1,
                                                    id2geo=id2geo)
    test_queries_2.extend(
        test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], num_test_query / 10, 1000, id2geo=id2geo))
    print("Sampling 2 edges validation queries")
    # val_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], 10, 900)
    val_queries_2 = test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], num_val_query * 9 / 10, 1,
                                                   id2geo=id2geo)
    val_queries_2.extend(
        test_graph.sample_test_queries(train_graph, ["2-chain", "2-inter"], num_val_query / 10, 1000, id2geo=id2geo))
    val_queries_2 = list(set(val_queries_2) - set(test_queries_2))
    print(len(val_queries_2))
    # Sampling 3 edges testing and validation queries
    print("Sampling 3 edges testing queries")
    test_queries_3 = test_graph.sample_test_queries(train_graph,
                                                    ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"],
                                                    num_test_query * 9 / 10, 1, id2geo=id2geo)
    test_queries_3.extend(
        test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"],
                                       num_test_query / 10, 1000, id2geo=id2geo))
    print("Sampling 3 edges validation queries")
    val_queries_3 = test_graph.sample_test_queries(train_graph,
                                                   ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"],
                                                   num_val_query * 9 / 10, 1, id2geo=id2geo)
    val_queries_3.extend(
        test_graph.sample_test_queries(train_graph, ["3-chain", "3-inter", "3-inter_chain", "3-chain_inter"],
                                       num_val_query / 10, 1000, id2geo=id2geo))
    val_queries_3 = list(set(val_queries_3) - set(test_queries_3))
    print(len(val_queries_3))
    print("Dumping 2/3 edges testing/validation queries")
    # pickle.dump([q.serialize() for q in test_queries_2], open(data_dir + "/test_queries_2-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump([q.serialize() for q in test_queries_3], open(data_dir + "/test_queries_3-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump([q.serialize() for q in val_queries_2], open(data_dir + "/val_queries_2-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    # pickle.dump([q.serialize() for q in val_queries_3], open(data_dir + "/val_queries_3-newclean.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""
    pickle.dump([q.serialize() for q in test_queries_2],
                open(data_dir + "/test_queries_2{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in test_queries_3],
                open(data_dir + "/test_queries_3{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_2],
                open(data_dir + "/val_queries_2{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in val_queries_3],
                open(data_dir + "/val_queries_3{}.pkl".format(file_postfix), "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def clean_test(train_queries, test_queries):
    """
    Delete queries in test_queries which also appear in the train_queries
    """
    for query_type in train_queries:
        train_set = set(train_queries[query_type])
        test_queries[query_type] = [q for q in test_queries[query_type] if not q in train_set]
    return test_queries


def parallel_sample_worker(pid, num_samples, graph, data_dir, is_test, test_edges, mp_result_dir=None, id2geo=None):
    """
    Args:
        id2geo: node id => [longitude, latitude] 
                if not None, we sample geographic query with target node as geographic entity
    """
    print("Running worker {}".format(pid))
    queries_2 = graph.sample_queries(2, num_samples, 100 if is_test else 1, verbose=True, id2geo=id2geo)
    queries_3 = graph.sample_queries(3, num_samples, 100 if is_test else 1, verbose=True, id2geo=id2geo)
    print("Done running worker, now saving data", pid)
    if mp_result_dir is None:
        mp_data_dir = data_dir
    else:
        mp_data_dir = mp_result_dir

    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""
    pickle.dump([q.serialize() for q in queries_2],
                open(mp_data_dir + "/queries_2-{:d}{:s}.pkl".format(pid, file_postfix), "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump([q.serialize() for q in queries_3],
                open(mp_data_dir + "/queries_3-{:d}{:s}.pkl".format(pid, file_postfix), "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)


def parallel_sample(graph, num_workers, samples_per_worker, data_dir, test=False, start_ind=None, mp_result_dir=None,
                    id2geo=None):
    """
    Use multiprocessing to sample queries
    Args:
        graph:
        num_workers:
        samples_per_worker: query samples per arity per worker
        data_dir:
        test: True/False
            True: remove the test and val triples from KG
        id2geo: node id => [longitude, latitude] 
                if not None, we sample geographic query with target node as geographic entity
    """
    if not test:
        print("Loading test/val data to remove them from train graph")
        test_edges = load_queries(data_dir + "/test_edges.pkl")
        val_edges = load_queries(data_dir + "/val_edges.pkl")
        print("Remove {} edges from the origin KG".format(len(test_edges + val_edges)))
        graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges + val_edges])
    else:
        test_edges = []
        val_edges = []
    proc_range = range(num_workers) if start_ind is None else range(start_ind, num_workers + start_ind)
    procs = [Process(target=parallel_sample_worker,
                     args=[i, samples_per_worker, graph, data_dir, test, val_edges + test_edges, mp_result_dir, id2geo])
             for i in proc_range]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    queries_2 = []
    queries_3 = []

    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""

    if mp_result_dir is None:
        mp_data_dir = data_dir
    else:
        mp_data_dir = mp_result_dir
    for i in range(num_workers):
        new_queries_2 = load_queries(mp_data_dir + "queries_2-{:d}{:s}.pkl".format(i, file_postfix), keep_graph=True)
        queries_2.extend(new_queries_2)
        new_queries_3 = load_queries(mp_data_dir + "queries_3-{:d}{:s}.pkl".format(i, file_postfix), keep_graph=True)
        queries_3.extend(new_queries_3)
    return queries_2, queries_3


def parallel_inter_query_sample_worker(pid, num_samples, graph, data_dir, is_test, max_inter_size=5, mp_result_dir=None,
                                       id2geo=None):
    """
    Args:
        id2geo: node id => [longitude, latitude] 
                if not None, we sample geographic query with target node as geographic entity
    """
    print("Running worker", pid)
    if mp_result_dir is None:
        mp_data_dir = data_dir
    else:
        mp_data_dir = mp_result_dir

    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""

    for arity in range(2, max_inter_size + 1):
        queries = graph.sample_inter_queries_by_arity(arity, num_samples, 100 if is_test else 1, verbose=True,
                                                      id2geo=id2geo)
        print("worker {:d}: saving {}-inter query".format(pid, arity))
        pickle.dump([q.serialize() for q in queries],
                    open(mp_data_dir + "queries_{:d}-{:d}{:s}.pkl".format(arity, pid, file_postfix), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

    print("Done running worker {:d}".format(pid))


def parallel_inter_query_sample(graph, num_workers, samples_per_worker, data_dir, max_inter_size=5, test=False,
                                start_ind=None, mp_result_dir=None, id2geo=None):
    """
    Use multiprocessing to sample queries
    Args:
        graph:
        num_workers:
        samples_per_worker: query samples per arity per worker
        data_dir:
        test: True/False
            True: remove the test and val triples from KG
        id2geo: node id => [longitude, latitude] 
                if not None, we sample geographic query with target node as geographic entity
    """
    if not test:
        print("Loading test/val data..")
        test_edges = load_queries(data_dir + "/test_edges.pkl")
        val_edges = load_queries(data_dir + "/val_edges.pkl")
        # I add this here
        print("Remove {} edges from the origin KG".format(len(test_edges + val_edges)))
        graph.remove_edges([(q.target_node, q.formula.rels[0], q.anchor_nodes[0]) for q in test_edges + val_edges])
    else:
        test_edges = []
        val_edges = []
    proc_range = range(num_workers) if start_ind is None else range(start_ind, num_workers + start_ind)
    procs = [Process(target=parallel_inter_query_sample_worker,
                     args=[i, samples_per_worker, graph, data_dir, test, max_inter_size, mp_result_dir, id2geo]) for i
             in proc_range]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    if mp_result_dir is None:
        mp_data_dir = data_dir
    else:
        mp_data_dir = mp_result_dir

    if id2geo is not None:
        file_postfix = "-geo"
    else:
        file_postfix = ""

    queries_dict = dict()
    for arity in range(2, max_inter_size + 1):
        queries = []
        for i in range(num_workers):
            new_queries = load_queries(mp_data_dir + "/queries_{:d}-{:d}{:s}.pkl".format(arity, i, file_postfix),
                                       keep_graph=True)
            queries.extend(new_queries)
        queries_dict[arity] = queries

    return queries_dict
