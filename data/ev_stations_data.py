import pandas as pd
import numpy as np

def generate_sample_ev_stations():
    """
    Generate sample EV charging station data combining Global and India datasets
    """
    
    global_stations = [
        {"station_id": "US001", "name": "Tesla Supercharger - Downtown LA", "latitude": 34.0522, "longitude": -118.2437, 
         "country": "USA", "city": "Los Angeles", "address": "123 Main St", "type": "Fast Charging", 
         "connectors": "CCS, CHAdeMO", "power_kw": 150, "availability": "Available", "pricing": 0.28, "rating": 4.5},
        
        {"station_id": "US002", "name": "ChargePoint Station - San Francisco", "latitude": 37.7749, "longitude": -122.4194,
         "country": "USA", "city": "San Francisco", "address": "456 Market St", "type": "Level 2", 
         "connectors": "J1772", "power_kw": 7.2, "availability": "Available", "pricing": 0.15, "rating": 4.2},
        
        {"station_id": "US003", "name": "EVgo Fast Charger - Seattle", "latitude": 47.6062, "longitude": -122.3321,
         "country": "USA", "city": "Seattle", "address": "789 Pike St", "type": "Fast Charging",
         "connectors": "CCS, CHAdeMO", "power_kw": 100, "availability": "In Use", "pricing": 0.32, "rating": 4.0},
        
        {"station_id": "UK001", "name": "BP Pulse - London", "latitude": 51.5074, "longitude": -0.1278,
         "country": "UK", "city": "London", "address": "10 Oxford St", "type": "Fast Charging",
         "connectors": "CCS, Type 2", "power_kw": 120, "availability": "Available", "pricing": 0.35, "rating": 4.3},
        
        {"station_id": "UK002", "name": "Ionity Charger - Manchester", "latitude": 53.4808, "longitude": -2.2426,
         "country": "UK", "city": "Manchester", "address": "25 Deansgate", "type": "Ultra Fast",
         "connectors": "CCS", "power_kw": 350, "availability": "Available", "pricing": 0.69, "rating": 4.7},
        
        {"station_id": "DE001", "name": "EnBW Charging - Berlin", "latitude": 52.5200, "longitude": 13.4050,
         "country": "Germany", "city": "Berlin", "address": "Unter den Linden 5", "type": "Fast Charging",
         "connectors": "CCS, Type 2", "power_kw": 150, "availability": "Available", "pricing": 0.45, "rating": 4.4},
    ]
    
    india_stations = [
        {"station_id": "IN001", "name": "Tata Power EZ Charge - Bangalore", "latitude": 12.9716, "longitude": 77.5946,
         "country": "India", "city": "Bangalore", "address": "MG Road, Bangalore", "type": "Fast Charging",
         "connectors": "CCS, CHAdeMO, Type 2", "power_kw": 60, "availability": "Available", "pricing": 0.10, "rating": 4.1},
        
        {"station_id": "IN002", "name": "Ather Grid - Delhi", "latitude": 28.7041, "longitude": 77.1025,
         "country": "India", "city": "Delhi", "address": "Connaught Place", "type": "Fast Charging",
         "connectors": "CCS, Type 2", "power_kw": 50, "availability": "Available", "pricing": 0.08, "rating": 4.3},
        
        {"station_id": "IN003", "name": "Fortum Charge & Drive - Mumbai", "latitude": 19.0760, "longitude": 72.8777,
         "country": "India", "city": "Mumbai", "address": "Bandra West", "type": "Fast Charging",
         "connectors": "CCS, CHAdeMO", "power_kw": 50, "availability": "In Use", "pricing": 0.09, "rating": 4.0},
        
        {"station_id": "IN004", "name": "Statiq EV Charger - Hyderabad", "latitude": 17.3850, "longitude": 78.4867,
         "country": "India", "city": "Hyderabad", "address": "Hi-Tech City", "type": "Level 2",
         "connectors": "Type 2", "power_kw": 7.2, "availability": "Available", "pricing": 0.06, "rating": 3.9},
        
        {"station_id": "IN005", "name": "ChargeZone - Chennai", "latitude": 13.0827, "longitude": 80.2707,
         "country": "India", "city": "Chennai", "address": "Anna Nagar", "type": "Fast Charging",
         "connectors": "CCS, Type 2", "power_kw": 60, "availability": "Available", "pricing": 0.10, "rating": 4.2},
        
        {"station_id": "IN006", "name": "Magenta ChargeGrid - Pune", "latitude": 18.5204, "longitude": 73.8567,
         "country": "India", "city": "Pune", "address": "Koregaon Park", "type": "Fast Charging",
         "connectors": "CCS, CHAdeMO, Type 2", "power_kw": 50, "availability": "Available", "pricing": 0.09, "rating": 4.4},
        
        {"station_id": "IN007", "name": "Revos EV Charging - Kolkata", "latitude": 22.5726, "longitude": 88.3639,
         "country": "India", "city": "Kolkata", "address": "Park Street", "type": "Level 2",
         "connectors": "Type 2", "power_kw": 7.2, "availability": "Maintenance", "pricing": 0.07, "rating": 3.8},
        
        {"station_id": "IN008", "name": "Tata Power - Jaipur", "latitude": 26.9124, "longitude": 75.7873,
         "country": "India", "city": "Jaipur", "address": "MI Road", "type": "Fast Charging",
         "connectors": "CCS, Type 2", "power_kw": 60, "availability": "Available", "pricing": 0.08, "rating": 4.1},
    ]
    
    all_stations = global_stations + india_stations
    
    df = pd.DataFrame(all_stations)
    
    df['opening_hours'] = '24/7'
    df['amenities'] = df.apply(lambda x: 
        np.random.choice(['WiFi, Restroom', 'Cafe, WiFi', 'Shopping Mall', 'Restaurant', 'Rest Area'], 1)[0], 
        axis=1)
    
    return df

def load_ev_stations():
    """
    Load and return EV charging stations dataset
    """
    return generate_sample_ev_stations()

def get_stations_by_country(country):
    """
    Filter stations by country
    """
    df = load_ev_stations()
    return df[df['country'] == country]

def get_stations_by_availability(availability_status='Available'):
    """
    Filter stations by availability status
    """
    df = load_ev_stations()
    return df[df['availability'] == availability_status]

def get_station_statistics():
    """
    Get summary statistics of charging stations
    """
    df = load_ev_stations()
    
    stats = {
        'total_stations': len(df),
        'countries': df['country'].nunique(),
        'cities': df['city'].nunique(),
        'avg_power_kw': df['power_kw'].mean(),
        'avg_rating': df['rating'].mean(),
        'avg_pricing': df['pricing'].mean(),
        'availability_distribution': df['availability'].value_counts().to_dict()
    }
    
    return stats
