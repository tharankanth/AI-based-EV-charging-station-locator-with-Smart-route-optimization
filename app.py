import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go

from data.ev_stations_data import load_ev_stations, get_station_statistics
from utils.data_processor import EVDataProcessor
from utils.route_optimizer import RouteOptimizer
from models.recommendation_model import StationRecommendationModel
from models.chatbot import EVChatbot

st.set_page_config(
    page_title="EV Charging Station Locator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load EV charging stations data"""
    return load_ev_stations()

@st.cache_resource
def initialize_models():
    """Initialize ML models and utilities"""
    processor = EVDataProcessor()
    optimizer = RouteOptimizer()
    recommender = StationRecommendationModel()
    chatbot = EVChatbot()
    return processor, optimizer, recommender, chatbot

def create_map(stations_df, center_lat=20.5937, center_lon=78.9629, zoom=4):
    """Create interactive map with charging stations"""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )
    
    for idx, row in stations_df.iterrows():
        color = 'green' if row['availability'] == 'Available' else 'red' if row['availability'] == 'Maintenance' else 'orange'
        
        icon = folium.Icon(color=color, icon='bolt', prefix='fa')
        
        popup_html = f"""
        <div style="width: 250px;">
            <h4>{row['name']}</h4>
            <p><b>Location:</b> {row['city']}, {row['country']}</p>
            <p><b>Address:</b> {row['address']}</p>
            <p><b>Power:</b> {row['power_kw']} kW</p>
            <p><b>Connectors:</b> {row['connectors']}</p>
            <p><b>Pricing:</b> ${row['pricing']}/kWh</p>
            <p><b>Rating:</b> {row['rating']} ⭐</p>
            <p><b>Status:</b> <span style="color: {color};">{row['availability']}</span></p>
            <p><b>Hours:</b> {row['opening_hours']}</p>
            <p><b>Amenities:</b> {row['amenities']}</p>
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row['name'],
            icon=icon
        ).add_to(m)
    
    if 'distance_km' in stations_df.columns and len(stations_df) > 0:
        nearest = stations_df.iloc[0]
        folium.Marker(
            location=[center_lat, center_lon],
            popup="Your Location",
            tooltip="You are here",
            icon=folium.Icon(color='blue', icon='user', prefix='fa')
        ).add_to(m)
        
        folium.PolyLine(
            locations=[[center_lat, center_lon], [nearest['latitude'], nearest['longitude']]],
            color='blue',
            weight=3,
            opacity=0.7,
            popup=f"Route to {nearest['name']}"
        ).add_to(m)
    
    return m

def main():
    """Main application"""
    
    stations_df = load_data()
    processor, optimizer, recommender, chatbot = initialize_models()
    
    st.title("⚡ EV Charging Station Locator")
    st.markdown("### AI/ML-Driven Smart Route Optimization for Electric Vehicles")
    
    st.sidebar.header("🔍 Search & Filter")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Station Finder", "📊 Analytics Dashboard", "💬 AI Assistant", "ℹ️ About"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Find Charging Stations Near You")
            
            location_input = st.text_input(
                "Enter your location (City, Country) or coordinates",
                value="Bangalore, India",
                help="e.g., 'New York, USA' or enter coordinates manually below"
            )
            
            coord_col1, coord_col2 = st.columns(2)
            with coord_col1:
                user_lat = st.number_input("Latitude", value=12.9716, format="%.4f")
            with coord_col2:
                user_lon = st.number_input("Longitude", value=77.5946, format="%.4f")
        
        with col2:
            st.subheader("Filters")
            
            countries = ['All'] + sorted(stations_df['country'].unique().tolist())
            selected_country = st.selectbox("Country", countries)
            
            availability_options = ['All'] + sorted(stations_df['availability'].unique().tolist())
            selected_availability = st.selectbox("Availability", availability_options)
            
            min_power = st.slider("Minimum Power (kW)", 0, 350, 50)
            
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
            
            max_distance = st.slider("Maximum Distance (km)", 10, 500, 100)
        
        filtered_df = stations_df.copy()
        
        if selected_country != 'All':
            filtered_df = filtered_df[filtered_df['country'] == selected_country]
        
        if selected_availability != 'All':
            filtered_df = filtered_df[filtered_df['availability'] == selected_availability]
        
        filtered_df = filtered_df[filtered_df['power_kw'] >= min_power]
        filtered_df = filtered_df[filtered_df['rating'] >= min_rating]
        
        nearby_stations = processor.find_nearest_stations(
            filtered_df, user_lat, user_lon, n=len(filtered_df)
        )
        
        nearby_stations = nearby_stations[nearby_stations['distance_km'] <= max_distance]
        
        nearby_stations = processor.rank_stations(nearby_stations)
        
        st.subheader(f"📍 Found {len(nearby_stations)} Charging Stations")
        
        if len(nearby_stations) > 0:
            map_obj = create_map(nearby_stations, user_lat, user_lon, zoom=6)
            folium_static(map_obj, width=1200, height=600)
            
            st.subheader("🎯 Recommended Stations (Ranked by Score)")
            
            for idx, row in nearby_stations.head(5).iterrows():
                with st.expander(f"🔋 {row['name']} - Score: {row['station_score']:.1f}/100"):
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Distance", f"{row['distance_km']:.2f} km")
                        st.metric("Est. Time", f"{row['estimated_time_min']:.0f} min")
                    
                    with col_b:
                        st.metric("Power", f"{row['power_kw']} kW")
                        st.metric("Pricing", f"${row['pricing']}/kWh")
                    
                    with col_c:
                        st.metric("Rating", f"{row['rating']} ⭐")
                        st.metric("Status", str(row['availability']))
                    
                    st.write(f"**Address:** {row['address']}, {row['city']}, {row['country']}")
                    st.write(f"**Connectors:** {row['connectors']}")
                    st.write(f"**Amenities:** {row['amenities']}")
                    st.write(f"**Hours:** {row['opening_hours']}")
                    
                    if st.button(f"Get Optimal Route to {row['name']}", key=f"route_{idx}"):
                        optimal_station, path_info = optimizer.find_optimal_station(
                            (user_lat, user_lon),
                            nearby_stations.head(10)
                        )
                        if path_info:
                            st.success(f"✅ Optimal route calculated!")
                            st.write(f"**Distance:** {path_info['distance_km']} km")
                            st.write(f"**Estimated Time:** {path_info['estimated_time_min']} minutes")
            
            st.subheader("📋 All Stations (Detailed View)")
            display_df = nearby_stations[['name', 'city', 'country', 'distance_km', 'estimated_time_min', 
                                         'power_kw', 'pricing', 'rating', 'availability', 'station_score']].copy()
            display_df.columns = ['Station Name', 'City', 'Country', 'Distance (km)', 'Time (min)', 
                                  'Power (kW)', 'Price ($/kWh)', 'Rating', 'Status', 'Score']
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.warning("No charging stations found matching your criteria. Try adjusting the filters.")
    
    with tab2:
        st.subheader("📊 Charging Station Analytics")
        
        stats = get_station_statistics()
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric("Total Stations", stats['total_stations'])
        with metric_col2:
            st.metric("Countries Covered", stats['countries'])
        with metric_col3:
            st.metric("Avg Power", f"{stats['avg_power_kw']:.1f} kW")
        with metric_col4:
            st.metric("Avg Rating", f"{stats['avg_rating']:.2f} ⭐")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.subheader("Stations by Country")
            country_counts = stations_df['country'].value_counts().reset_index()
            country_counts.columns = ['Country', 'Count']
            fig1 = px.bar(country_counts, x='Country', y='Count', 
                         color='Count', color_continuous_scale='viridis',
                         title="Distribution of Charging Stations by Country")
            st.plotly_chart(fig1, use_container_width=True)
        
        with chart_col2:
            st.subheader("Charging Power Distribution")
            fig2 = px.histogram(stations_df, x='power_kw', nbins=20,
                               title="Distribution of Charging Power (kW)",
                               labels={'power_kw': 'Power (kW)', 'count': 'Number of Stations'})
            st.plotly_chart(fig2, use_container_width=True)
        
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            st.subheader("Availability Status")
            availability_counts = stations_df['availability'].value_counts().reset_index()
            availability_counts.columns = ['Status', 'Count']
            fig3 = px.pie(availability_counts, values='Count', names='Status',
                         title="Station Availability Status",
                         color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig3, use_container_width=True)
        
        with chart_col4:
            st.subheader("Pricing vs Power")
            fig4 = px.scatter(stations_df, x='power_kw', y='pricing', 
                            size='rating', color='type',
                            hover_data=['name', 'city'],
                            title="Charging Power vs Pricing",
                            labels={'power_kw': 'Power (kW)', 'pricing': 'Price ($/kWh)'})
            st.plotly_chart(fig4, use_container_width=True)
        
        st.subheader("🌍 Geographic Distribution")
        fig5 = px.scatter_geo(stations_df, 
                             lat='latitude', 
                             lon='longitude',
                             hover_name='name',
                             hover_data={'latitude': False, 'longitude': False, 
                                        'city': True, 'power_kw': True, 'rating': True},
                             color='country',
                             size='power_kw',
                             title="Global EV Charging Station Network")
        st.plotly_chart(fig5, use_container_width=True)
        
        st.subheader("📈 Station Ratings Analysis")
        fig6 = px.box(stations_df, x='type', y='rating', color='type',
                     title="Rating Distribution by Charging Type",
                     labels={'type': 'Charging Type', 'rating': 'Rating'})
        st.plotly_chart(fig6, use_container_width=True)
    
    with tab3:
        st.subheader("💬 AI Chatbot Assistant")
        st.markdown("Ask me anything about EV charging stations, pricing, route planning, or general EV information!")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        user_question = st.chat_input("Ask me about EV charging...")
        
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            with st.chat_message("user"):
                st.markdown(user_question)
            
            response = chatbot.get_response(user_question)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            with st.chat_message("assistant"):
                st.markdown(response)
        
        with st.sidebar:
            st.subheader("💡 Quick Questions")
            if st.button("Find nearest station"):
                st.session_state.chat_history.append({
                    "role": "user", 
                    "content": "How do I find the nearest charging station?"
                })
                st.rerun()
            
            if st.button("Charging costs"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "What are typical charging costs?"
                })
                st.rerun()
            
            if st.button("Connector types"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "What are the different connector types?"
                })
                st.rerun()
            
            if st.button("Charging time"):
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": "How long does it take to charge?"
                })
                st.rerun()
            
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab4:
        st.subheader("ℹ️ About This Project")
        
        st.markdown("""
        ### EV Charging Station Locator with Smart Route Optimization
        
        This application leverages **Artificial Intelligence** and **Machine Learning** to provide intelligent 
        EV charging station recommendations and route optimization.
        
        #### 🎯 Key Features:
        
        - **Interactive Map Visualization**: Browse charging stations on an interactive map with real-time availability
        - **Smart Route Optimization**: Uses Dijkstra's algorithm to find the shortest path to charging stations
        - **ML-Based Recommendations**: Machine learning models rank stations based on distance, power, pricing, and ratings
        - **AI Chatbot**: NLP-powered chatbot answers questions about EV charging
        - **Advanced Filtering**: Filter stations by country, availability, power, rating, and distance
        - **Data Analytics**: Comprehensive visualizations showing station distribution, pricing trends, and more
        
        #### 🔧 Technology Stack:
        
        - **Framework**: Streamlit
        - **Data Processing**: Pandas, NumPy
        - **Machine Learning**: Scikit-learn (KNN, Random Forest)
        - **Route Optimization**: NetworkX (Graph algorithms)
        - **Geospatial Analysis**: Geopy, Folium
        - **Visualizations**: Plotly, Folium
        - **Natural Language Processing**: NLTK
        
        #### 📊 Datasets Integrated:
        
        - Global EV Charging Stations Dataset
        - Electric Vehicle Charging Stations in India
        - Custom EV-specific chatbot training data
        
        #### 🚀 ML/AI Capabilities:
        
        1. **K-Nearest Neighbors (KNN)**: Recommends stations based on user preferences
        2. **Route Optimization**: Dijkstra's algorithm for shortest path calculation
        3. **Scoring System**: Multi-factor scoring considering distance, power, price, rating, and availability
        4. **NLP Chatbot**: Intent recognition and response generation for user queries
        
        #### 📈 Performance Metrics:
        
        - Real-time distance calculation using geodesic measurements
        - Estimated travel time based on average speeds
        - Station ranking with weighted scoring (0-100 scale)
        - Route efficiency calculation
        
        #### 👨‍💻 Development:
        
        Built with professional coding standards following modular architecture:
        - `data/`: Dataset management and loading
        - `models/`: ML models and AI chatbot
        - `utils/`: Data processing and route optimization utilities
        - `app.py`: Main Streamlit application
        
        ---
        
        **Version**: 1.0.0  
        **Last Updated**: November 2025
        """)
        
        st.info("💡 **Tip**: Use the sidebar filters to customize your search and find the perfect charging station for your needs!")

if __name__ == "__main__":
    main()
