# EV Charging Station Locator

AI/ML-Driven Smart Route Optimization for Electric Vehicles

## Overview

An intelligent web application that helps EV owners find nearby charging stations with optimized routes using machine learning and advanced algorithms.

## Features

- **Interactive Station Finder**: Real-time map showing charging stations with availability status
- **Smart Route Optimization**: Dijkstra's algorithm for shortest path calculation
- **ML-Powered Recommendations**: KNN-based station ranking considering distance, power, pricing, and ratings
- **AI Chatbot Assistant**: NLP-powered chatbot for EV charging queries
- **Advanced Filtering**: Filter by country, availability, power, rating, and distance
- **Analytics Dashboard**: Comprehensive visualizations of station data and trends

## Technology Stack

- **Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (KNN, Random Forest)
- **Route Optimization**: NetworkX (Graph algorithms)
- **Geospatial**: Geopy, Folium
- **Visualizations**: Plotly
- **NLP**: NLTK

## Project Structure

```
├── app.py                          # Main Streamlit application
├── data/
│   ├── ev_stations_data.py        # Dataset management
│   └── __init__.py
├── models/
│   ├── chatbot.py                 # AI chatbot with NLP
│   ├── recommendation_model.py    # ML recommendation engine
│   └── __init__.py
├── utils/
│   ├── data_processor.py          # Data preprocessing utilities
│   ├── route_optimizer.py         # Route optimization algorithms
│   └── __init__.py
└── .streamlit/
    └── config.toml                # Streamlit configuration
```

## Datasets

- Global EV Charging Stations Dataset
- Electric Vehicle Charging Stations in India
- Custom EV chatbot training data

## ML/AI Components

1. **K-Nearest Neighbors**: Station recommendations based on user preferences
2. **Dijkstra's Algorithm**: Shortest path route optimization
3. **Multi-Factor Scoring**: Weighted scoring system (distance, power, price, rating, availability)
4. **NLP Intent Recognition**: Pattern-based chatbot for user queries

## Key Capabilities

- Distance calculation using geodesic measurements
- Real-time station filtering and ranking
- Interactive map visualization with color-coded availability
- Estimated travel time based on distance
- Comprehensive analytics with multiple visualizations
- Conversational AI for EV charging information

## Application Interface

The application features four main sections:

1. **Station Finder**: Interactive map with location-based search and station recommendations
2. **Analytics Dashboard**: Data visualizations showing station distribution, pricing trends, and usage patterns
3. **AI Assistant**: Conversational chatbot for EV charging questions
4. **About**: Project information and technical details

---
