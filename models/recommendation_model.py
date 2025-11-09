import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class StationRecommendationModel:
    """
    Machine Learning model for intelligent charging station recommendations
    """
    
    def __init__(self):
        self.knn_model = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_features(self, stations_df):
        """
        Prepare feature matrix for ML models
        """
        features = stations_df[['latitude', 'longitude', 'power_kw', 'pricing', 'rating']].copy()
        
        if 'distance_km' in stations_df.columns:
            features['distance_km'] = stations_df['distance_km']
        else:
            features['distance_km'] = 0
        
        features = features.fillna(0)
        
        return features
    
    def train_knn_model(self, stations_df):
        """
        Train K-Nearest Neighbors model for station recommendations
        """
        features = self.prepare_features(stations_df)
        
        features_scaled = self.scaler.fit_transform(features)
        
        self.knn_model.fit(features_scaled)
        self.is_trained = True
        
        return True
    
    def recommend_stations_knn(self, user_preferences, stations_df, n_recommendations=5):
        """
        Recommend stations using KNN based on user preferences
        """
        if not self.is_trained:
            self.train_knn_model(stations_df)
        
        user_feature_vector = np.array([[
            user_preferences.get('latitude', 0),
            user_preferences.get('longitude', 0),
            user_preferences.get('preferred_power_kw', 100),
            user_preferences.get('max_pricing', 0.5),
            user_preferences.get('min_rating', 4.0),
            user_preferences.get('max_distance_km', 50)
        ]])
        
        user_feature_scaled = self.scaler.transform(user_feature_vector)
        
        distances, indices = self.knn_model.kneighbors(user_feature_scaled, n_neighbors=n_recommendations)
        
        recommended_stations = stations_df.iloc[indices[0]].copy()
        recommended_stations['recommendation_score'] = 100 - (distances[0] / distances[0].max() * 100)
        
        return recommended_stations
    
    def predict_station_suitability(self, station_row, user_preferences):
        """
        Predict how suitable a station is for user based on preferences
        """
        suitability_score = 0
        
        if 'distance_km' in station_row:
            max_distance = user_preferences.get('max_distance_km', 100)
            if station_row['distance_km'] <= max_distance:
                suitability_score += 30
        
        if station_row['power_kw'] >= user_preferences.get('preferred_power_kw', 50):
            suitability_score += 25
        
        if station_row['pricing'] <= user_preferences.get('max_pricing', 1.0):
            suitability_score += 20
        
        if station_row['rating'] >= user_preferences.get('min_rating', 3.5):
            suitability_score += 15
        
        if station_row['availability'] == 'Available':
            suitability_score += 10
        
        return min(100, suitability_score)
    
    def rank_by_preferences(self, stations_df, user_preferences):
        """
        Rank stations based on user preferences
        """
        df = stations_df.copy()
        
        df['suitability_score'] = df.apply(
            lambda row: self.predict_station_suitability(row, user_preferences),
            axis=1
        )
        
        df = df.sort_values('suitability_score', ascending=False)
        
        return df
    
    def get_personalized_recommendations(self, user_location, user_preferences, stations_df, n=5):
        """
        Get personalized station recommendations combining multiple factors
        """
        from utils.data_processor import EVDataProcessor
        
        processor = EVDataProcessor()
        
        stations_with_distance = processor.find_nearest_stations(
            stations_df, 
            user_location[0], 
            user_location[1],
            n=len(stations_df)
        )
        
        ranked_stations = self.rank_by_preferences(stations_with_distance, user_preferences)
        
        top_recommendations = ranked_stations.head(n)
        
        return top_recommendations
    
    def analyze_user_pattern(self, historical_data):
        """
        Analyze user charging patterns from historical data
        """
        if len(historical_data) == 0:
            return {
                'preferred_power_range': '50-150 kW',
                'preferred_pricing_range': '$0.10-$0.50',
                'avg_distance_traveled': 'N/A',
                'most_used_connector': 'CCS'
            }
        
        df = pd.DataFrame(historical_data)
        
        pattern_analysis = {
            'preferred_power_range': f"{df['power_kw'].quantile(0.25):.0f}-{df['power_kw'].quantile(0.75):.0f} kW",
            'preferred_pricing_range': f"${df['pricing'].quantile(0.25):.2f}-${df['pricing'].quantile(0.75):.2f}",
            'avg_distance_traveled': f"{df['distance_km'].mean():.2f} km",
            'most_used_connector': df['connectors'].mode()[0] if len(df) > 0 else 'CCS'
        }
        
        return pattern_analysis
