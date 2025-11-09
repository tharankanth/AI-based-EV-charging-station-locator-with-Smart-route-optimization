# EV Charging Station Locator

## Overview

This is a web-based EV (Electric Vehicle) Charging Station Locator application built with Streamlit. The application helps users find nearby charging stations, provides intelligent recommendations using machine learning, and offers an AI-powered chatbot for user queries. It features an interactive map interface, route optimization capabilities, and detailed analytics about charging station availability, pricing, and ratings across global and Indian markets.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Framework**: Streamlit web application framework
- **Rationale**: Streamlit provides rapid development of data-driven web applications with minimal frontend code. It's ideal for prototyping and deploying ML/data science applications quickly.
- **Layout**: Wide layout with expandable sidebar for filters and controls
- **Visualization Libraries**: 
  - Folium for interactive maps with marker clusters
  - Plotly for interactive charts and graphs
  - streamlit-folium for Streamlit-Folium integration

**Alternatives Considered**: Flask/Django with React frontend
- **Pros of Streamlit**: Faster development, built-in caching, native support for data visualization
- **Cons**: Limited customization compared to full-stack frameworks, less control over UI/UX

### Backend Architecture

**Core Components**:

1. **Data Layer** (`data/ev_stations_data.py`)
   - Generates and manages sample EV charging station data
   - Supports both global (USA, UK, Germany) and India-specific stations
   - In-memory data storage using pandas DataFrames
   - **Rationale**: Sample data generation allows for demonstration without requiring external data sources

2. **Data Processing** (`utils/data_processor.py`)
   - Feature: Geodesic distance calculations using geopy
   - Feature: Data normalization with scikit-learn StandardScaler
   - Feature: Label encoding for categorical variables
   - **Rationale**: Centralized data processing ensures consistency across the application and prepares data for ML models

3. **Route Optimization** (`utils/route_optimizer.py`)
   - Algorithm: NetworkX graph-based routing with Dijkstra's shortest path
   - Creates graph network from station locations
   - **Rationale**: Graph algorithms provide optimal route planning between user location and charging stations
   - **Trade-off**: In-memory graph construction vs. pre-computed routes (chose flexibility over performance)

4. **Machine Learning Models**:

   **Station Recommendation** (`models/recommendation_model.py`)
   - Algorithm: K-Nearest Neighbors (KNN) for location-based recommendations
   - Algorithm: Random Forest Classifier for preference-based filtering
   - Feature engineering: latitude, longitude, power capacity, pricing, rating, distance
   - **Rationale**: KNN excels at spatial similarity; Random Forest handles complex feature interactions
   - **Pros**: Simple, interpretable, works well with small datasets
   - **Cons**: Requires retraining when data changes significantly

   **AI Chatbot** (`models/chatbot.py`)
   - NLP Framework: NLTK for tokenization and text processing
   - Pattern-matching based intent recognition
   - Pre-defined intents: greeting, find_station, pricing, connector_types, etc.
   - **Rationale**: Rule-based approach provides predictable, controllable responses without requiring large language models
   - **Alternatives**: Transformer-based models (GPT, BERT) would provide more natural responses but require more resources

### Data Storage Solutions

**Current Implementation**: In-memory pandas DataFrames
- **Rationale**: Suitable for prototype/demo with sample data
- **Limitation**: Data doesn't persist between sessions
- **Future Consideration**: The architecture is designed to easily integrate with PostgreSQL/SQLite for production use

**Data Schema**:
- Station attributes: station_id, name, latitude, longitude, country, city, address
- Technical specs: type, connectors, power_kw
- Operational data: availability, pricing, rating
- Computed fields: distance_km, estimated_time_min

### Caching Strategy

**Streamlit Caching**:
- `@st.cache_data`: For data loading functions (stations data)
- `@st.cache_resource`: For ML model initialization
- **Rationale**: Prevents redundant computations and model reloading on each user interaction, significantly improving performance

### Key Design Patterns

1. **Separation of Concerns**: Clear separation between data layer, processing utilities, ML models, and presentation
2. **Factory Pattern**: Model initialization through `initialize_models()` function
3. **Strategy Pattern**: Multiple recommendation algorithms (KNN, Random Forest) can be selected based on use case

## External Dependencies

### Core Libraries

**Data Processing**:
- `pandas`: DataFrame operations and data manipulation
- `numpy`: Numerical computations and array operations

**Geospatial**:
- `geopy`: Geodesic distance calculations between coordinates
- `folium`: Interactive map generation with OpenStreetMap tiles
- `streamlit-folium`: Integration layer between Streamlit and Folium

**Visualization**:
- `plotly`: Interactive charts (express and graph_objects modules)
- Both high-level (px) and low-level (go) APIs used for different visualization needs

**Machine Learning**:
- `scikit-learn`: 
  - RandomForestClassifier for classification tasks
  - NearestNeighbors for KNN-based recommendations
  - StandardScaler for feature normalization
  - LabelEncoder for categorical encoding
- `networkx`: Graph algorithms for route optimization

**Natural Language Processing**:
- `nltk`: Tokenization, stopword removal, text preprocessing
- Required NLTK data: punkt tokenizer, stopwords corpus, punkt_tab

**Web Framework**:
- `streamlit`: Web application framework and UI components

### Map Tiles

- **OpenStreetMap**: Free, open-source map tiles via Folium
- **Alternative**: Could integrate commercial providers (Mapbox, Google Maps) for enhanced features

### Future Integration Points

The architecture supports easy integration with:
- PostgreSQL/MySQL for persistent data storage
- Real-time charging station APIs (ChargePoint, Tesla, etc.)
- Payment gateways for booking functionality
- User authentication services
- Google Maps/Mapbox for enhanced routing