import networkx as nx
import numpy as np
from geopy.distance import geodesic

class RouteOptimizer:
    """
    Route optimization using graph algorithms (Dijkstra's algorithm)
    """
    
    def __init__(self):
        self.graph = nx.Graph()
        
    def build_graph_from_stations(self, stations_df, user_location):
        """
        Build a graph network from charging stations
        """
        self.graph.clear()
        
        user_node = 'USER'
        self.graph.add_node(user_node, 
                           latitude=user_location[0], 
                           longitude=user_location[1],
                           type='user')
        
        for idx, row in stations_df.iterrows():
            station_node = row['station_id']
            self.graph.add_node(station_node,
                              latitude=row['latitude'],
                              longitude=row['longitude'],
                              name=row['name'],
                              type='station',
                              power_kw=row['power_kw'],
                              availability=row['availability'])
        
        for node1 in self.graph.nodes():
            for node2 in self.graph.nodes():
                if node1 != node2:
                    lat1 = self.graph.nodes[node1]['latitude']
                    lon1 = self.graph.nodes[node1]['longitude']
                    lat2 = self.graph.nodes[node2]['latitude']
                    lon2 = self.graph.nodes[node2]['longitude']
                    
                    distance = geodesic((lat1, lon1), (lat2, lon2)).kilometers
                    
                    self.graph.add_edge(node1, node2, weight=distance)
        
        return self.graph
    
    def find_shortest_path(self, start_node, end_node):
        """
        Find shortest path using Dijkstra's algorithm
        """
        try:
            path = nx.dijkstra_path(self.graph, start_node, end_node, weight='weight')
            path_length = nx.dijkstra_path_length(self.graph, start_node, end_node, weight='weight')
            
            return {
                'path': path,
                'distance_km': round(path_length, 2),
                'estimated_time_min': round((path_length / 60) * 60, 2)
            }
        except nx.NetworkXNoPath:
            return None
    
    def find_optimal_station(self, user_location, stations_df):
        """
        Find the optimal charging station based on shortest path
        """
        self.build_graph_from_stations(stations_df, user_location)
        
        best_station = None
        best_distance = float('inf')
        best_path_info = None
        
        for idx, row in stations_df.iterrows():
            if row['availability'] == 'Available':
                station_id = row['station_id']
                path_info = self.find_shortest_path('USER', station_id)
                
                if path_info and path_info['distance_km'] < best_distance:
                    best_distance = path_info['distance_km']
                    best_station = row
                    best_path_info = path_info
        
        return best_station, best_path_info
    
    def calculate_multi_stop_route(self, user_location, stations_df, num_stops=3):
        """
        Calculate optimal route visiting multiple charging stations
        """
        self.build_graph_from_stations(stations_df, user_location)
        
        available_stations = stations_df[stations_df['availability'] == 'Available']
        
        if len(available_stations) < num_stops:
            num_stops = len(available_stations)
        
        station_ids = available_stations['station_id'].tolist()[:num_stops]
        
        route = ['USER'] + station_ids
        total_distance = 0
        
        for i in range(len(route) - 1):
            path_info = self.find_shortest_path(route[i], route[i+1])
            if path_info:
                total_distance += path_info['distance_km']
        
        return {
            'route': route,
            'total_distance_km': round(total_distance, 2),
            'estimated_time_min': round((total_distance / 60) * 60, 2),
            'num_stations': num_stops
        }
    
    def get_alternative_routes(self, user_location, stations_df, target_station_id, k=3):
        """
        Get k alternative routes to a target station
        """
        self.build_graph_from_stations(stations_df, user_location)
        
        try:
            paths = list(nx.shortest_simple_paths(self.graph, 'USER', target_station_id, weight='weight'))
            
            alternative_routes = []
            for i, path in enumerate(paths[:k]):
                path_length = 0.0
                for j in range(len(path)-1):
                    edge_data = self.graph[path[j]][path[j+1]]
                    path_length += float(edge_data['weight'])
                
                alternative_routes.append({
                    'route_number': i + 1,
                    'path': path,
                    'distance_km': round(path_length, 2),
                    'estimated_time_min': round((path_length / 60) * 60, 2)
                })
            
            return alternative_routes
        except:
            return []
    
    def calculate_route_efficiency(self, distance_km, power_kw, pricing):
        """
        Calculate route efficiency score considering distance, charging power, and cost
        """
        distance_efficiency = max(0, 100 - distance_km)
        power_efficiency = min(100, (power_kw / 150) * 100)
        cost_efficiency = max(0, 100 - (pricing * 100))
        
        overall_efficiency = (
            distance_efficiency * 0.50 +
            power_efficiency * 0.30 +
            cost_efficiency * 0.20
        )
        
        return round(overall_efficiency, 2)
