import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
import seaborn as sns
import os
import osmnx as ox
import datetime
import networkx as nx
import json
import math
import numpy as np

# Data collection functions
#------------------------------------------------------------------------------

def get_data():
    '''
    Function to put taxi_data into a dataframe from csv and return itself.

    Returns: taxi_data (pd.DataFrame)

    '''
    parent_folder = os.path.dirname(os.getcwd())
    data_location = os.path.join(parent_folder, 'VI. Data', 'train.csv')
    taxi_data = pd.read_csv(data_location)
    return taxi_data

def get_graph():
    '''
    Returns a graph of road network in porto in the form of a networkx multidigraph
    '''
    porto_graph_full = ox.graph_from_place('Porto, Portugal', which_result=2, network_type='drive_service',simplify=False, truncate_by_edge=True)
    # Now simplify the graph to get rid of unneeded intermediary nodes
    #porto_graph = ox.simplify_graph(porto_graph_full, strict=True)
    # now make sure isolated nodes are all removed

    #porto_graph = ox.core.remove_isolated_nodes(porto_graph)
    # now simply graph to get rid of nodes that correspond to dead_ends
    #node_degree_dict = ox.utils.count_streets_per_node(porto_graph)
    #dead_end_nodes = []
    #for key, value in node_degree_dict.items():
     #   if value == 1:
     #       dead_end_nodes.append(key)
    # Now let's drop those nodes
    #porto_graph.remove_nodes_from(dead_end_nodes)

    return porto_graph_full

def return_links_of_interest(graph):
    '''
    Due to the graph network coming straight from osm having nodes at every bend
    in the road, there are far too many roads. Moreover, many roads are quite short
    and so will almost always take less than 15 seconds to traverse. Considering that
    our measurements are spaced 15 seconds apart, we only want to learn the pdfs
    for roads that take around or more than 15 seconds to traverse as we can't exatly
    be sure how long it took to traverse shorter roads. Thus, we only consider modelling
    a subset of the links of the network known as 'links of interest'.

    Parameters:
    ----------
    Graph (networkx multidigraph): Graph that you want to get the links_of_interest from

    Returns:
    --------
    links_of_interest (dictionary): mapping from these 'links of interest' to their corresponding
                                    indices (just an integer from 0-len(links_of_interest)).
    '''

    # speed limit on inner city roads is 50 mph in Porto.
    # We want the time taken on links of interest to be around 15 seconds
    # to reduce our measurement error.
    # Therefore the minimum length of road we are interested in, assuming
    # a min cutoff time of 10 seconds and an average speed of
    # 30 mph is (30*1.6)/3600)*10*1000 ~= 130m
    # use 100m just to be safe
    # This get's rid of roughly half the edges

    #we want to use the simplified graph as regular graph has too many edges
    simple_graph = ox.simplify_graph(graph, strict=True)
    cutoff = 100
    edges = list(simple_graph.edges(data='length'))
    edges_of_interest = {}
    i=0
    for edge_data in edges:
        edge_name = get_edge_name(edge_data[0],edge_data[1])
        edge_length = edge_data[2]
        if edge_length > cutoff:
            #edge[0] and edge[1] correspond to the nodes
            if edge_name not in edges_of_interest:
                # edges are repeated in edges for some reason
                edges_of_interest[edge_name] = i
                i+=1

    return edges_of_interest


def return_edges_in_graph(graph):
    '''
    Function to return a list of edges in the graph to make sure that map_matcher() maps
    GPS coordinates to roads actually in my network. Basically just a helper function.


    Parameters:
    -----------
    graph (networkx multidigraph): Graph we want to get list of edges from.

    Returns:
    --------
    edges (list): list of edges in the graph named the normal way (node1-node2)
    '''
    edge_data = list(graph.edges(data=True))
    edges = []
    for edge in edge_data:
        #edge[0] and edge[1] correspond to the nodes
        edges.append(get_edge_name(edge[0],edge[1]))

    return edges



def return_edge_data(graph):
    '''
    Function that maps edge names (links) to their associted lengths. Helper function.
    Can be modified to return more information such as the link shapefiles.

    Parameters:
    -----------
    graph (networkx multidigraph): Graph we want to get edge data from

    Returns:
    --------
    edge_data (Dicitonary): Dictionary that maps edges to their lengths
    '''
    edges = list(graph.edges(data='length'))
    edge_data ={}
    for i,data in enumerate(edges):
        edge_name = get_edge_name(data[0],data[1])
        edge_length = data[2]
        edge_data[edge_name] = {'length': edge_length}

    return edge_data

def get_edge_name(u, v):
    '''
    Returns name of an edge in the form node_u-node_v for reference purposes. Helper function.

    Parameters:
    -----------
    u (string): Name of first node
    v (string): Name of second node

    Returns:
    --------
    edge_name (string): The resulting name of the edge made from u and v

    '''
    edge_name = str(u) + '-' + str(v)
    return edge_name

def return_node_data(graph):
    '''
    Returns a dictionary of key:values where the keys represent each node in the graph
    and the values are each a list of long,lat pairs for that node. Helper function.

    This is done for easy data access in other functions, as a list (the default data format for the nodes)
    is not suitable.

    Parameters:
    -----------
    graph (networkx multidigraph): Graph we want to get node data from

    Returns:
    ---------
    node_dict (dictionary): Dictionary that maps nodes to their GPS locations in
                            the form {node:{'long':long_val, 'lat': lat_val}}
    '''
    node_dict = {}
    node_data = graph.nodes(data=True)
    # Now go through list of nodes and put them into disct
    for node in node_data:
        node_dict[node[0]] = {'long':node[1]['x'], 'lat': node[1]['y']}
    return node_dict

def simple_link_mapper(graph, strict=True):
    """
    CREDIT: https://github.com/gboeing/osmnx/blob/master/osmnx/simplify.py

    ------- modified for different functionality -------

    The default graph extracted has lots of links due to every bend in the road
    being classified as a link. Due to graph having lots of links (and the map matcher requiring it to do so),
    we need a way to map these smaller links to their simplified ones so we can
    model the longer roads as desired. Helper function.

    Returns a dictionary unsimplified edges to their corresponding simplified links
    as well as the proportion (lengths) of those simplified links they represent.

    Parameters:
    ----------
    graph (networkx multidigraph)
    strict (bool): if False, allow nodes to be end points even if they fail all other rules
                   but have edges with different OSM IDs
    Returns:
    -------
    link_mapper (dictionary): Dictionary that maps unsimplified edges to their corresponding simplified links
                              as well as the proportion (lengths) of those simplified links they represent.
    """

    # first identify all the nodes that are endpoints
    endpoints = set([node for node in graph.nodes() if ox.is_endpoint(graph, node, strict=strict)])
    paths_to_simplify = []
  # for each endpoint node, look at each of its successor nodes
    for node in endpoints:
        for successor in graph.successors(node):
            if successor not in endpoints:
                # if the successor is not an endpoint, build a path from the
                # endpoint node to the next endpoint node
                try:
                    path = ox.build_path(graph, successor, endpoints, path=[node, successor])
                    paths_to_simplify.append(path)
                except RuntimeError:
                    continue
            else:
                # just add node, successor into paths to simplify anyways (so it get's added to link_mapper later)
                paths_to_simplify.append([node,successor])
    # Now we have a list of paths to simplify
    # we will now use the edges of a pre-simplified graph for comparison
    simplified_graph = ox.simplify_graph(graph, strict=True)
    simple_edge_data = return_edge_data(simplified_graph)
    #also get the edge data for normal graph
    edge_data = return_edge_data(graph)
    link_mapper = {}
    # now iterate through the paths to simplify
    for path in paths_to_simplify:
        # simplified edge corresponds to first and last of path
        simple_edge = get_edge_name(path[0],path[-1])
        simple_edge_length = simple_edge_data[simple_edge]['length']
        # now iterate through interstitial nodes
        for u, v in zip(path[:-1], path[1:]):
            edge_name_1 = get_edge_name(u,v)
            # now take into account other direction too
            edge_name_2 = get_edge_name(v,u)
            # now find what proportion of simple link length this link is
            proportion = edge_data[edge_name_1]['length']/simple_edge_length
            link_mapper[edge_name_1] = {'simple_link': simple_edge, 'proportion':proportion}
            link_mapper[edge_name_2] = {'simple_link': simple_edge, 'proportion':proportion}


    return link_mapper





# Dataframe processing functions
#------------------------------------------------------------------------------

def convert_datetime(df):
    '''
    Function that uses the unix timestamp column in data to generate
    corresponding column with timestamps as pandas DateTime objects.


    Parameters:
    ------------
    df (pandas dataframe): Dataframe to be modified

    Returns:
    --------
    df: dataframe with added 'date_time' column
    '''
    df['date_time'] = pd.to_datetime(df['TIMESTAMP'],unit='s')
    return df

def add_timestamps(trip):
    '''
    Pandas function (acts on the dataframe) that adds timestamps to every link in a
    trip's link sequence. I.e. adds 15 second spaced timestamps for each link traversed
    in the corresponding interval. Helper function.

    This is done so that instead of having data in a trip by trip format, we can get get all the link
    sequence data across all trips and sort by timestamp. The kalman filter will then operate on this data.

    Parameters:
    ------------
    trip (pandas series): Trip to be modified

    Returns:
    --------
    trip (pandas series): Row of trip_data is returned with the partial_link_sequence having
                          timestamps added.
    '''
    partial_link_sequence = eval(trip.link_sequence)
    current_timestamp = trip.TIMESTAMP
    for index1, timestep in enumerate(partial_link_sequence):
        # now go through each link visited in that timestep and add timestamps
        for index2, link in enumerate(timestep):
            # link is in tuple format which isn't useful as tuple is immutable so have to change to list
            # SHOULD HAVE DONE AT THE START (when making link sequence column)
            # BUT NOW DOCKER ISSUES REMAIN SO WILL HAVE TO DO IT MESSY WAY
            new_link_data = list(link)
            new_link_data.append(current_timestamp)
            partial_link_sequence[index1][index2] = new_link_data
        current_timestamp+=15
    trip.link_sequence = partial_link_sequence
    return trip


def transform_into_kalman_input(trip_dataframe):
    '''
    The train/test data after adding the partial_link_sequence column is in a trip by trip basis.
    However, in reality, the taxi dispatcher will receive GPS pings from any taxi in the fleet
    as soon as it is available. Therefore, the trip by trip format of our data is not suitable and
    we must transform it into a data structure that amalgamates all links traversed data across all trips
    and store it in a format that is sorted by the time at which each link traversion was reported.

    Then, in the kalman filter, we can choose a discretization time as required and simply move along this list,
    using only the links traversed that were reported in this discretization time as the 'observed measurements'
    for our update equations

    Parameters:
    ------------
    trip data dataframe (pandas dataframe): Trip_data with link_sequence column included

    Returns:
    --------
    sorted_reported_data (list of lists): (sorted by timestamp reported) which each sublist having the
                                          format (link traversed, proption of link traversed,
                                          timestamp when reported)

    '''

    # let's sort the training data by date
    trip_data = trip_dataframe.sort_values('date_time')

    # let's add the timestamps to all links traversed (when they were reported)
    trip_data = trip_data.apply(add_timestamps, axis=1)

    # let's flatten our trip by trip data to almagamate all links reported by all taxis
    link_sequence_list = trip_data['link_sequence'].tolist()
    flat_link_sequence_list = []
    for trip in link_sequence_list:
        for timestep in trip:
            flat_link_sequence_list.append(timestep)

    # now need to sort this list of lists into a sequence of links traversed by timestamp of when it was reported
    sorted_reported_data = sorted(flat_link_sequence_list, key=lambda x: x[0][2])
    return sorted_reported_data



def map_matcher(longitude, latitude, edges_in_graph):
    '''
    Using a routing engine I found online and run locally using docker, this function
    takes a GPS coordinate and maps it to a link in the network.

    Parameters:
    -----------
    longitude (float): longitude of GPS coordinate
    latitude (float): latitude of GPS coordinate

    Returns:
    --------
    (Array): [First_node of link, second_node of link, Noisy GPS input mapped onto closest point on nearest link]

    '''
    # using routing engine that I run locally using docker
    # get 30 nearest roads in-case the closest one is not a drivable road for some reason (should be drivable)
    url = 'http://127.0.0.1:5000/nearest/v1/car/{},{}?number=30'.format(longitude,latitude)
    response = requests.get(url)
    data = response.json()
    for match in data['waypoints']:
        nodes = match['nodes']
        # location is the GPS mapped onto the road
        location = match['location']
        link_name = get_edge_name(nodes[0], nodes[1])
        if link_name in edges_in_graph:
            return nodes[0], nodes[1], location
        else:
            continue
    # if we haven't found a match, the most likely reason is the taxi left the area that
    # our graph is defined in. There is nothing we can do at this point so we will return
    # a False and the partial link sequence returned by get_partial_link_sequence will
    #correspond to GPS coordinates taken up to this point
    return False, False, False


def get_partial_link_sequence(trip, graph, link_mapper):
    '''
    Function that takes a trip's GPS polyline vector and trasforms it into
    a list of corresponding links travelled as well as proportion of the link travelled.
    Uses most of the functions defined above.

    Parameters:
    -----------
    trip (pandas series): trip data
    graph (networkx multidigraph): road network of interest
    link_mapper (dictionary): see simple_link_mapper()

    Returns:
    --------
    simple_partial_link_sequence (array): sequence of simplified links traversed by the taxi for that trip

    '''
    gps_trip_list = json.loads(trip.POLYLINE)
    node_data = return_node_data(graph)
    edge_data = return_edge_data(graph)
    edges_in_graph = return_edges_in_graph(graph)
    # initialise array to store our partial link sequence
    partial_link_sequence = []
    # start off the algorithm with the first set of GPS points
    start_long = gps_trip_list[0][0]
    start_lat = gps_trip_list[0][1]
    prev_node_1, prev_node_2, prev_location = map_matcher(start_long, start_lat, edges_in_graph)
    if prev_node_1 == False:
        # means map_matching has failed
        return partial_link_sequence
    prev_link_dist = edge_data[get_edge_name(prev_node_1,prev_node_2)]['length']
    for gps_pair in gps_trip_list[1:]:
        longitude = gps_pair[0]
        latitude = gps_pair[1]
        # find where we are now
        current_node_1, current_node_2, current_location = map_matcher(longitude, latitude, edges_in_graph)
        if current_node_1 == False:
            # means map_matching has failed
            return return_simple_partial_link_sequence(partial_link_sequence, link_mapper)
        # if we're still on the same link:
        if (current_node_1,current_node_2) == (prev_node_1, prev_node_2):
            lat1 = prev_location[1]
            lng1 = prev_location[0]
            lat2 = current_location[1]
            lng2 = current_location[0]
            distance_travelled = ox.great_circle_vec(lat1, lng1, lat2, lng2)
            # work out proportion of road travelled
            alpha = distance_travelled/prev_link_dist
            partial_link_sequence.append([(get_edge_name(current_node_1, current_node_2),alpha)])
            # now just update previous values to be current ones in preparation for next GPS point
            prev_node_1, prev_node_2, prev_location = current_node_1, current_node_2, current_location
        elif set([current_node_1, current_node_2]) != set([prev_node_1,prev_node_2]):
            # i.e. we are on a completely new non-adjacent link to the previous link
            link_sequence = []
            alpha = []
            # first work out how much we had to travel to leave previous link
            lat1 = prev_location[1]
            lng1 = prev_location[0]
            lat2 = node_data[prev_node_2]['lat']
            lng2 = node_data[prev_node_2]['long']
            lat3 = node_data[prev_node_1]['lat']
            lng3 = node_data[prev_node_1]['long']
            lat4 = current_location[1]
            lng4 = current_location[0]
            lat5 = node_data[current_node_1]['lat']
            lng5 = node_data[current_node_1]['long']
            # we don't know if the car left through prev_node_2 or prev_node_1
            # so we will estimate this by saying that the closest node to the current
            # location is the node that the taxi passed out the previous link from
            if ox.great_circle_vec(lat4, lng4, lat2, lng2) < ox.great_circle_vec(lat4, lng4, lat3, lng3):
                # means we went through prev_node_2
                distance_travelled_prev_link = ox.great_circle_vec(lat1, lng1, lat2, lng2)
                alpha.append(distance_travelled_prev_link/prev_link_dist)
                link_sequence.append(get_edge_name(prev_node_1, prev_node_2))
                # work out links that must have been traversed from prev_node_2 to current_node_1
                try:
                    route=nx.shortest_path(graph,prev_node_2,current_node_1, weight='length')
                except nx.NetworkXNoPath:
                    # happens sometimes when we can't find a path - don't know why
                    route = []
                if len(route)>0:
                    for node_index in range(0,len(route)-1):
                        link = get_edge_name(route[node_index], route[node_index + 1])
                        link_sequence.append(link)
                        alpha.append(1)

                # Now deal with final link that's partially traversed
                link_sequence.append(get_edge_name(current_node_1, current_node_2))
                current_link_dist = edge_data[get_edge_name(current_node_1, current_node_2)]['length']
                distance_travelled = ox.great_circle_vec(lat4, lng4, lat5, lng5)
                alpha.append(distance_travelled/current_link_dist)
                partial_link_sequence.append(list(zip(link_sequence,alpha)))
                # prev values update
                prev_node_1, prev_node_2, prev_location = current_node_1, current_node_2, current_location
                prev_link_dist = current_link_dist
            else:
                # means we went through prev_node_1
                distance_travelled_prev_link = ox.great_circle_vec(lat1, lng1, lat3, lng3)
                alpha.append(distance_travelled_prev_link/prev_link_dist)
                link_sequence.append(get_edge_name(prev_node_1, prev_node_2))
                # work out links that must have been traversed from prev_node_1 to current_node_1
                try:
                    route=nx.shortest_path(graph,prev_node_2,current_node_1, weight='length')
                except nx.NetworkXNoPath:
                    # happens sometimes when we can't find a path - don't know why
                    route = []
                if len(route)>0:
                    for node_index in range(0,len(route)-1):
                        link = get_edge_name(route[node_index], route[node_index + 1])
                        link_sequence.append(link)
                        alpha.append(1)

                # Now deal with final link that's partially traversed
                link_sequence.append(get_edge_name(current_node_1, current_node_2))
                current_link_dist = edge_data[get_edge_name(current_node_1, current_node_2)]['length']
                distance_travelled = ox.great_circle_vec(lat4, lng4, lat5, lng5)
                alpha.append(distance_travelled/current_link_dist)
                # prev values update
                prev_node_1, prev_node_2, prev_location = current_node_1, current_node_2, current_location
                prev_link_dist = current_link_dist

        else:
            # This means that car must now be on adjacent link to
            # previous link with common node current_node_1
            alpha = []
            # first calculate how much road was left to travel on previous link
            lat1 = prev_location[1]
            lng1 = prev_location[0]
            lat2 = node_data[current_node_1]['lat']
            lng2 = node_data[current_node_1]['long']
            distance_travelled_prev_link = ox.great_circle_vec(lat1, lng1, lat2, lng2)
            alpha.append(distance_travelled_prev_link/prev_link_dist)
            # now add how much was travelled in current link
            lat1 = current_location[1]
            lng1 = current_location[0]
            lat2 = node_data[current_node_1]['lat']
            lng2 = node_data[current_node_1]['long']
            current_link_dist = edge_data[get_edge_name(current_node_1,current_node_2)]['length']
            distance_travelled_current_link = ox.great_circle_vec(lat1, lng1, lat2, lng2)
            alpha.append(distance_travelled_current_link/current_link_dist)
            # now add these two to the partial link sequence
            partial_link_sequence.append([(get_edge_name(prev_node_1, prev_node_2), alpha[0]),(get_edge_name(current_node_1, current_node_2), alpha[1])])
            # update prev values
            prev_node_1, prev_node_2, prev_location = current_node_1, current_node_2, current_location
            prev_link_dist = current_link_dist


    # now turn this partial link sequence into partial link sequence of the simple links of network
    simple_partial_link_sequence = return_simple_partial_link_sequence(partial_link_sequence, link_mapper)


    return simple_partial_link_sequence

def return_simple_partial_link_sequence(partial_link_sequence, link_mapper):
    '''
    Partial link sequences from the get_partial_link_sequence algorithm
    are by default in terms of the non-simplified links of the network.

    This function turns the partial link sequence into a link sequence made up of the simplified links
    of the network with all the alphas scaled accordingly.

    Parameters:
    -----------
    partial_link_sequence: sequence of unsimplified links traversed by the taxi for that trip
    link_mapper (dictionary): see simple_link_mapper()

    Returns:
    --------
    simple_partial_link_sequence (array): partial link sequence made up of simple links of the network
    '''
    simple_partial_link_sequence = []
    for timestep in partial_link_sequence:
        simple_sequence = []
        for link in timestep:
            simple_link = link_mapper[link[0]]['simple_link']
            alpha = link_mapper[link[0]]['proportion']*link[1]
            simple_sequence.append((simple_link,alpha))
        simple_partial_link_sequence.append(simple_sequence)
    return simple_partial_link_sequence




# Kalman Filtering Functions
#------------------------------------------------------------------------------
def kalman_predict(X,P,A,W=[],B=[],U=[]):
    '''
    Function that returns predicted values of X and P using input matrices.
    '''
    X_predicted = np.dot(A, X) + np.dot(B, U)
    P_predicted = np.dot(A, np.dot(P, A.T))
    return X_predicted, P_predicted

def kalman_update(X, P, Y, H, N):
    '''
    Function that updates X and P using measurment related matrices Y,H and N.
    '''
    predicted_means = np.dot(H,X)
    predicted_covariance = N + np.dot(H, np.dot(P,H.T))
    kalman_gain = np.dot(P, np.dot(H.T, np.linalg.inv(predicted_covariance)))
    #print('X', X.shape)
    #print('Y', Y.shape)
    #print('kalman gain', kalman_gain.shape)
    #print('predicted means', predicted_means.shape)
    #print('Y - predicted means',(Y[:,0] - predicted_means[:,0]).reshape((-1, 1)).shape)
    X_updated = X + np.dot(kalman_gain, (Y[:,0] - predicted_means[:,0]).reshape((-1, 1)))
    #print('X_updated', X_updated.shape)
    P_updated = P - np.dot(kalman_gain, np.dot(predicted_covariance, kalman_gain.T))
    return X_updated, P_updated



def make_measurement_matrices(last_index, last_timestamp, sorted_reported_data, links_of_interest):
    '''
    Function that makes Y, H and N matrices for the Measurement Update step of the Kalman Filter for each discretisation
    step.

    Parameters:
    ------------
    last_index (integer): The index of the last measurement (in sorted_reported_data) that was used in
                          the previous block. This is just to speed the function up so the measurements
                          don't have searched from the start.

    last_timestamp (integer):  The timestamp of the last measurment used in the last  discretisation block.
                               Measurements used in the current block have to be atleast within discret_time
                               from last_timestamp.

    sorted_reported_data (multidimensional array): The training data you are using which is the output of
                                                   passing train_data to the transform_into_kalman_input
                                                   function.

    links_of_interest (dictionary): Dictionary holding the links of interest and their corresponding indices in
                                    the state vector.


    Returns:
    ---------
    Y (1D numpy array): A vector of 15's whose size depends on the number of measurments seen in the current
                        block.

    H (2D numpy array): A matrix with each row corresponding to one measurment. The values in the columns represent
                        the total proportion of length of that column's particular link we have seen traversal
                        for in the current measurement (this can be greater than 1 due to multiple taxi measurments).

    N (2D numpy array): A diagonal matrix depresenting the measurment noise matrix. Each diagonal element is equal
                        to the variance of the measurment noise (The MNV variable below).


    next_start_index (integer): Index of the last measurment (in sorted_reported_data), used in the current block

    last_timestamp (integer): timestamp that the next discretisation block should use as its last_timestamp.

    new_week (Boolean): We want to know if a new week has been reached so that the performance measures can look at
                        the right bit of the test data - this boolean keeps track of that.

    '''
    #define measurement noise std to be 2 seconds
    MNstd = 2
    MNV = MNstd*MNstd
    state_size = len(links_of_interest.keys())
    size_y = 0
    # let's pool all measurments taken over a discretization time of 5 minutes = 300 seconds
    discret_time = 300
    H_as_list = []
    # also hold a boolean to see if any measurements have been made in next discretization step
    measurement_received = False
    # hold another boolean to say if we have reached measurements outside the next discretization step
    finished = False
    # hold another boolean that we return to say if we reached a new week (for plotting purposes)
    new_week = False
    # start right at where the last reporting was in the last discretization step
    for index, reporting in enumerate(sorted_reported_data[last_index+1:]):
        # let's initialise our row for H
        row = [0]*state_size
        for link in reporting:
            link_name = link[0]
            alpha = link[1]
            timestamp = link[2]
            # check if in discretization time
            if ((timestamp - last_timestamp) <0):
                break

            elif ((timestamp - last_timestamp) < 300):
                # check if it is a link of interest
                if link_name in links_of_interest:
                    # get the index of the link to fill in row with alpha
                    link_index = links_of_interest[link_name]
                    row[link_index] += alpha
                    measurement_received = True
            # since we have data spanning over tuesdays over multiple weeks, we have to check to see if our
            # data has moved to the next week, or no reporting will be identified as belonging to the next
            # discretization time's measurements.

            # thus, check if the timestamp of the current reporting is sufficiently larger than last_timestamp
            # to classify it as belonging to next week's data
            elif (timestamp - last_timestamp) > 300000:
                print('\n'+ '***************Went to next week**********'+'\n')
                # we need to shift the timestamp all the way to where the current reporting is now as otherwise
                # we will spend many 5 minute cycles with no measurements recorded - leading to large l-2 errors.
                last_timestamp = timestamp
                finished=True
                new_week=True
                break

            else:
                # we have gone through all measurements happening in next 5 min period
                finished=True
                # shift counter along by 5 minutes
                last_timestamp = last_timestamp + 300
                break
        # add row to the matrix
        H_as_list.append(row)
        size_y +=1
        if finished == True:
            break

        elif (measurement_received == False) and ((timestamp - last_timestamp) > 300):
            print('Got nothing in this block')
            # meaning we didn't get anything in next 5 min step
            # turn H into numpy array
            H = np.array(H_as_list)
            # make measurment noise variance matrix
            N = np.eye(size_y, dtype=int)*MNV
            # make Y which is just array of 0
            Y = np.full((size_y, 1), 0)
            # increment index counter to next measurement
            next_start_index = last_index + 1

            return Y, H, N, next_start_index, last_timestamp, new_week


        else:
            # go on to the next reporting
            continue

    # turn H into numpy array
    H = np.array(H_as_list)
    # make measurment noise variance matrix
    N = np.eye(size_y, dtype=int)*MNV
    # make Y which is just array of 15's
    Y = np.full((size_y, 1), 15)
    next_start_index = last_index + index

    return Y, H, N, next_start_index, last_timestamp, new_week



def measure_performance(sorted_test_data, last_timestamp, predicted_X,links_of_interest):
    '''
    This is a function for estimation performance testing. First, we gets all the measurements in the test data that
    happen in same 5 min block to the block we have predicted states for. Then, we identify which road the
    measurement was on, and what proportion of the road was travelled. We then map this to our state vector and
    multiply our predicted link travel time by the proportion travelled. If more than one link was travelled,
    we sum the predicted link travel times. We then calculate the absolute value of the error between the time
    stated by the measurment for the partial link (or multiple link traversal) (always 15) with our predicted time.
    We then calculate the average for the block to give an L1-error.

    Parameters:
    -----------

    sorted_test_data (multidimensional list): Output of test_data when fed to the transform_into_kalman_input function.

    last_timestamp (integer): Last timestamp from the training set  that is going to was used in the last measurement
                              update. The measurments we look at from the test set for performance testing should be
                              less than 300 seconds below this value.


    predicted_X (1D numpy vector): Current predicted state vector.

    Returns:
    --------

    performance (float): A number representing the average L1-error for the current block's estimation
                         or prediction.

    '''
    # set min_timestamp so estimation performance is done for all measurements
    # that happen in the time block we have predicted  for.
    min_timestamp = last_timestamp - 300

    sum_of_abs = 0
    num_measurements = 0
    for index, reporting in enumerate(sorted_test_data):
        # each reporting corresponds to 15 seconds
        # as the reporting can have multiple links, we'll have a variable that holds the sum of the total
        # predicted travel times (from predicted_X and the corresponding alphas measured)
        sum_of_predicted = 0
        for link in reporting:
            # first check if the link happened in the block we're interested in
            link_name = link[0]
            alpha = link[1]
            timestamp = link[2]
            if timestamp <= min_timestamp:
                # this measurment is not of concern
                break
            # need to check also if we've reached end of test data
            elif (timestamp > (min_timestamp + 300)) or (reporting == sorted_test_data[-1]):
                # means we've finished looking at block we're interest in, so we're good to return performance
                # measure now
                if num_measurements != 0:
                    performance = sum_of_abs/num_measurements
                else:
                    # if we didn't pick up any measurements
                    performance = 0
                return float(performance)
            else:
                # means we're in the block of interest
                # check if it is a link of interest
                if link_name in links_of_interest:
                    # get the index of the link so we can compare it to the the corresponding row of predicted_X
                    link_index = links_of_interest[link_name]
                    predicted_time_taken = alpha*predicted_X[link_index]
                    sum_of_predicted+=predicted_time_taken
        # Now I can calculate the error for current reporting
        # need to check if we actually picked up any measurements in current reporting:
        if sum_of_predicted != 0:
            #print('Compared 15 to {}'.format(sum_of_predicted))
            abs_of_error = abs(15 - sum_of_predicted)
            sum_of_abs+=abs_of_error
            num_measurements+=1
