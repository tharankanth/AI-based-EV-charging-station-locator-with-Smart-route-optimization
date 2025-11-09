import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class EVDataProcessor:
    """
    Data preprocessing and feature engineering for EV charging stations
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two coordinates using geodesic distance
        """
        try:
            point1 = (lat1, lon1)
            point2 = (lat2, lon2)
            distance_km = geodesic(point1, point2).kilometers
            return round(distance_km, 2)
        except:
            return None
    
    def calculate_estimated_time(self, distance_km, speed_kmh=60):
        """
        Calculate estimated travel time based on distance
        """
        if distance_km is None or distance_km == 0:
            return 0
        time_hours = distance_km / speed_kmh
        time_minutes = time_hours * 60
        return round(time_minutes, 2)
    
    def find_nearest_stations(self, stations_df, user_lat, user_lon, n=5):
        """
        Find n nearest charging stations to user location
        """
        stations_df = stations_df.copy()
        
        stations_df['distance_km'] = stations_df.apply(
            lambda row: self.calculate_distance(user_lat, user_lon, row['latitude'], row['longitude']),
            axis=1
        )
        
        stations_df['estimated_time_min'] = stations_df['distance_km'].apply(
            lambda d: self.calculate_estimated_time(d)
        )
        
        nearest_stations = stations_df.nsmallest(n, 'distance_km')
        
        return nearest_stations
    
    def filter_by_power(self, stations_df, min_power_kw=50):
        """
        Filter stations by minimum charging power
        """
        return stations_df[stations_df['power_kw'] >= min_power_kw]
    
    def filter_by_availability(self, stations_df, status='Available'):
        """
        Filter stations by availability status
        """
        return stations_df[stations_df['availability'] == status]
    
    def filter_by_rating(self, stations_df, min_rating=4.0):
        """
        Filter stations by minimum rating
        """
        return stations_df[stations_df['rating'] >= min_rating]
    
    def preprocess_for_ml(self, stations_df):
        """
        Preprocess data for machine learning models
        """
        df = stations_df.copy()
        
        categorical_columns = ['type', 'availability', 'country']
        numerical_columns = ['power_kw', 'pricing', 'rating']
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        if len(df) > 1:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        
        return df
    
    def calculate_station_score(self, row):
        """
        Calculate overall score for a charging station based on multiple factors
        """
        distance_score = max(0, 100 - (row.get('distance_km', 0) * 2))
        
        power_score = min(100, (row['power_kw'] / 150) * 100)
        
        rating_score = (row['rating'] / 5) * 100
        
        availability_score = 100 if row['availability'] == 'Available' else 20
        
        pricing_score = max(0, 100 - (row['pricing'] * 100))
        
        total_score = (
            distance_score * 0.30 +
            power_score * 0.20 +
            rating_score * 0.20 +
            availability_score * 0.20 +
            pricing_score * 0.10
        )
        
        return round(total_score, 2)
    
    def rank_stations(self, stations_df):
        """
        Rank stations based on calculated scores
        """
        df = stations_df.copy()
        df['station_score'] = df.apply(self.calculate_station_score, axis=1)
        df = df.sort_values('station_score', ascending=False)
        return df
    
    def get_route_summary(self, station_row):
        """
        Generate route summary for a station
        """
        summary = {
            'station_name': station_row['name'],
            'distance_km': station_row.get('distance_km', 'N/A'),
            'estimated_time_min': station_row.get('estimated_time_min', 'N/A'),
            'address': station_row['address'],
            'city': station_row['city'],
            'power_kw': station_row['power_kw'],
            'connectors': station_row['connectors'],
            'pricing': station_row['pricing'],
            'rating': station_row['rating'],
            'availability': station_row['availability']
        }
        return summary
