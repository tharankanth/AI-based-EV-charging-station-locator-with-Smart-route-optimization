import random
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

class EVChatbot:
    """
    AI-powered chatbot for EV charging station queries using NLP
    """
    
    def __init__(self):
        self.intents = self._load_intents()
        self.stop_words = set(stopwords.words('english'))
        
    def _load_intents(self):
        """
        Load chatbot intents and responses for EV-related queries
        """
        intents = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'],
                'responses': [
                    "Hello! I'm here to help you with EV charging station information. How can I assist you today?",
                    "Hi there! Looking for charging stations or have questions about EV charging? I'm here to help!",
                    "Greetings! I can help you find charging stations, understand pricing, or answer any EV-related questions."
                ]
            },
            'find_station': {
                'patterns': ['find station', 'charging station', 'where to charge', 'nearest station', 
                           'charging point', 'find charger', 'locate station', 'station near me'],
                'responses': [
                    "I can help you find nearby charging stations! Please use the location finder on the main page to enter your current location, and I'll show you the nearest available stations.",
                    "To find charging stations near you, simply enter your location in the search box above. You'll see all nearby stations with distances and availability.",
                    "Looking for a charging station? Use the map interface to enter your location, and the system will display all nearby charging points with real-time availability."
                ]
            },
            'pricing': {
                'patterns': ['price', 'cost', 'how much', 'pricing', 'charging cost', 'fee', 'rate', 'expensive'],
                'responses': [
                    "Charging prices vary by station and location. Typical rates range from $0.06 to $0.69 per kWh. Fast chargers (50-150 kW) usually cost $0.20-$0.50 per kWh, while ultra-fast chargers (350 kW) can be higher.",
                    "EV charging costs depend on the station type: Level 2 chargers cost $0.06-$0.15 per kWh, Fast chargers cost $0.20-$0.35 per kWh, and Ultra-Fast chargers cost $0.40-$0.69 per kWh.",
                    "Pricing varies by location and charging speed. In India, rates are typically ₹7-₹10 per kWh. In the US and Europe, expect $0.20-$0.50 per kWh for fast charging."
                ]
            },
            'charging_time': {
                'patterns': ['how long', 'charging time', 'duration', 'time to charge', 'fast charging', 'quick charge'],
                'responses': [
                    "Charging time depends on your vehicle's battery capacity and the charger's power: Level 2 (7 kW) takes 4-8 hours for a full charge, Fast chargers (50-150 kW) take 30-60 minutes for 80%, Ultra-fast chargers (350 kW) can charge to 80% in 15-20 minutes.",
                    "Fast chargers (50-150 kW) typically add 100-200 km of range in about 30 minutes. Ultra-fast chargers (350 kW) can provide the same range in just 10-15 minutes!",
                    "For a typical EV with a 60 kWh battery: Level 2 charger takes 6-8 hours, 50 kW fast charger takes about 1 hour, 150 kW fast charger takes 25-30 minutes, and 350 kW ultra-fast takes 15-20 minutes for 80% charge."
                ]
            },
            'connector_types': {
                'patterns': ['connector', 'plug', 'cable', 'ccs', 'chademo', 'type 2', 'j1772', 'connector type'],
                'responses': [
                    "Common EV connectors include: CCS (Combined Charging System) for fast DC charging, CHAdeMO for fast DC charging (mainly Japanese EVs), Type 2 for AC charging in Europe/India, and J1772 for AC charging in North America.",
                    "The main connector types are: CCS (most modern EVs), CHAdeMO (Nissan, Mitsubishi), Type 2/Mennekes (European standard), and J1772 (North American standard). Most modern stations offer multiple connector types.",
                    "Your EV's connector depends on the manufacturer: Tesla uses their proprietary connector (or CCS in newer models), most European/Asian EVs use CCS or Type 2, Nissan Leaf uses CHAdeMO, and older EVs might use J1772 for AC charging."
                ]
            },
            'availability': {
                'patterns': ['available', 'availability', 'station status', 'in use', 'occupied', 'free', 'open'],
                'responses': [
                    "Station availability is shown in real-time on the map. Green markers indicate available stations, yellow shows 'in use', and red indicates maintenance or offline status.",
                    "You can check station availability on the interactive map. Each station shows its current status: Available, In Use, or Under Maintenance. We update this information regularly.",
                    "Real-time availability is displayed for each station. Filter by 'Available' status to see only stations you can use right now!"
                ]
            },
            'route_planning': {
                'patterns': ['route', 'plan trip', 'navigation', 'directions', 'best route', 'optimize route'],
                'responses': [
                    "Our route optimization feature uses advanced algorithms to find the shortest path to charging stations. Enter your destination, and the system will calculate the optimal route with estimated time and distance.",
                    "The route planner uses Dijkstra's algorithm to find the most efficient path to your selected charging station, considering distance, traffic patterns, and charging time.",
                    "For route planning, select your destination station from the map. The system will display the optimized route with distance, estimated travel time, and charging details."
                ]
            },
            'payment': {
                'patterns': ['payment', 'pay', 'credit card', 'payment method', 'how to pay', 'app', 'membership'],
                'responses': [
                    "Payment methods vary by charging network. Most stations accept: Mobile apps (network-specific), Credit/debit cards via touchscreen, RFID cards for members, and contactless payment. Some require pre-registration with the network.",
                    "You can typically pay using: 1) The charging network's mobile app, 2) Credit/debit card at the station, 3) RFID membership cards, or 4) Contactless payment (Apple Pay, Google Pay). Check individual station details for accepted methods.",
                    "Major charging networks like ChargePoint, EVgo, and Tesla Supercharger have their own apps for payment. Most stations also accept credit cards directly. Some networks offer monthly memberships for discounted rates."
                ]
            },
            'benefits': {
                'patterns': ['benefits', 'why ev', 'advantages', 'eco friendly', 'environment', 'savings'],
                'responses': [
                    "EV charging benefits include: Lower fuel costs (electricity is cheaper than gasoline), Environmental benefits (zero emissions), Convenience (charge at home overnight), Government incentives and rebates, and Lower maintenance costs.",
                    "EVs offer significant savings! Electricity costs about 1/3 of gasoline per kilometer. Plus, EVs have fewer moving parts, meaning less maintenance. Many regions also offer tax credits and incentives for EV owners.",
                    "Environmental and economic benefits of EVs: Zero tailpipe emissions, 60-70% lower running costs, Quiet operation, Instant torque and smooth acceleration, and Growing charging infrastructure worldwide."
                ]
            },
            'range_anxiety': {
                'patterns': ['range', 'range anxiety', 'battery life', 'how far', 'distance', 'run out'],
                'responses': [
                    "Modern EVs have ranges of 250-500+ km on a single charge. With our station locator, you can plan routes and find charging stations along your journey to eliminate range anxiety.",
                    "Range anxiety is becoming less of an issue! Most EVs now offer 300+ km of range, and fast charging networks are expanding rapidly. Our app helps you plan trips with charging stops along the way.",
                    "To manage range: Plan routes using our optimizer, Charge to 80% at fast chargers (faster than 100%), Use regenerative braking, and Keep your battery between 20-80% for daily use. The system will alert you to nearby stations!"
                ]
            },
            'help': {
                'patterns': ['help', 'support', 'assist', 'guide', 'how to use', 'tutorial'],
                'responses': [
                    "I can help you with: Finding nearby charging stations, Understanding pricing and charging times, Route planning and optimization, Connector types and compatibility, and General EV charging questions. What would you like to know?",
                    "Here's what I can assist with: 🔍 Locating stations near you, 💰 Pricing information, ⚡ Charging speeds and times, 🗺️ Route optimization, 📱 Payment methods, and 🔌 Connector compatibility. How can I help?",
                    "Need assistance? I can provide information on: Station locations and availability, Charging costs and times, Route planning features, Connector types, Payment methods, and EV benefits. Ask me anything!"
                ]
            },
            'thanks': {
                'patterns': ['thank', 'thanks', 'appreciate', 'helpful'],
                'responses': [
                    "You're welcome! Feel free to ask if you have more questions about EV charging.",
                    "Happy to help! Enjoy your EV charging experience!",
                    "My pleasure! Drive safe and charge smart!"
                ]
            }
        }
        
        return intents
    
    def preprocess_text(self, text):
        """
        Preprocess user input text
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        try:
            tokens = word_tokenize(text)
            tokens = [word for word in tokens if word not in self.stop_words]
        except:
            tokens = text.split()
        
        return tokens
    
    def match_intent(self, user_input):
        """
        Match user input to an intent using pattern matching
        """
        tokens = self.preprocess_text(user_input)
        user_text = ' '.join(tokens)
        
        best_match = None
        best_score = 0
        
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data['patterns']:
                pattern_tokens = self.preprocess_text(pattern)
                pattern_text = ' '.join(pattern_tokens)
                
                if pattern_text in user_text or user_text in pattern_text:
                    score = len(set(tokens) & set(pattern_tokens))
                    if score > best_score:
                        best_score = score
                        best_match = intent_name
        
        return best_match if best_score > 0 else 'default'
    
    def get_response(self, user_input):
        """
        Get chatbot response based on user input
        """
        intent = self.match_intent(user_input)
        
        if intent in self.intents:
            response = random.choice(self.intents[intent]['responses'])
        else:
            response = "I'm here to help with EV charging stations! You can ask me about finding stations, pricing, charging times, connectors, or route planning. What would you like to know?"
        
        return response
    
    def get_station_info(self, station_data):
        """
        Provide detailed information about a specific station
        """
        info = f"""
**Station Details:**
- **Name:** {station_data.get('name', 'N/A')}
- **Location:** {station_data.get('address', 'N/A')}, {station_data.get('city', 'N/A')}
- **Power:** {station_data.get('power_kw', 'N/A')} kW
- **Connectors:** {station_data.get('connectors', 'N/A')}
- **Pricing:** ${station_data.get('pricing', 'N/A')} per kWh
- **Rating:** {station_data.get('rating', 'N/A')} ⭐
- **Availability:** {station_data.get('availability', 'N/A')}
- **Hours:** {station_data.get('opening_hours', '24/7')}
"""
        return info.strip()
