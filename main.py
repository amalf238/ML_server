import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# Import your exact ML model classes from the notebook
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SearchRequest(BaseModel):
    description: str
    location: str  # Changed from latitude/longitude to location name

class WorkerResponse(BaseModel):
    worker_id: str
    worker_name: str
    service_type: str
    rating: float
    experience_years: int
    daily_wage_lkr: int
    phone_number: str
    city: str
    distance_km: float
    ai_confidence: float
    bio: str

class SearchResponse(BaseModel):
    workers: List[WorkerResponse]
    ai_analysis: Dict

# Location Name to Coordinates Converter
class LocationConverter:
    """Convert location names to coordinates"""
    
    def __init__(self):
        # Comprehensive Sri Lankan location database
        self.location_database = {
            # Major Cities
            'colombo': (6.9271, 79.8612),
            'kandy': (7.2906, 80.6337),
            'galle': (6.0535, 80.2210),
            'negombo': (7.2084, 79.8380),
            'jaffna': (9.6615, 80.0255),
            'kurunegala': (7.4818, 80.3609),
            'anuradhapura': (8.3114, 80.4037),
            'matara': (5.9549, 80.5550),
            'ratnapura': (6.6828, 80.3992),
            'trincomalee': (8.5874, 81.2152),
            'batticaloa': (7.7310, 81.6747),
            'badulla': (6.9934, 81.0550),
            'nuwara eliya': (6.9497, 80.7891),
            'ampara': (7.2978, 81.6722),
            'vavuniya': (8.7542, 80.4982),
            'mannar': (8.9810, 79.9044),
            'polonnaruwa': (7.9403, 81.0188),
            'hambantota': (6.1429, 81.1212),
            'puttalam': (8.0362, 79.8283),
            'kegalle': (7.2523, 80.3436),
            'monaragala': (6.8728, 81.3507),
            'kilinochchi': (9.3961, 80.3990),
            'mullativu': (9.2671, 80.8142),
            
            # Colombo Suburbs
            'koswatta': (6.8875, 79.8800),
            'dehiwala': (6.8560, 79.8638),
            'mount lavinia': (6.8374, 79.8634),
            'moratuwa': (6.7730, 79.8816),
            'kotte': (6.8905, 79.9015),
            'sri jayawardenepura kotte': (6.8905, 79.9015),
            'nugegoda': (6.8649, 79.8997),
            'maharagama': (6.8482, 79.9298),
            'rajagiriya': (6.9084, 79.8916),
            'battaramulla': (6.8992, 79.9186),
            'malabe': (6.9147, 79.9731),
            'kaduwela': (6.9381, 79.9897),
            'pelawatta': (6.8461, 79.9062),
            'thalawathugoda': (6.8738, 79.9750),
            'homagama': (6.8441, 80.0022),
            'kottawa': (6.8207, 79.9097),
            'piliyandala': (6.8008, 79.9226),
            'boralesgamuwa': (6.8417, 79.9025),
            'athurugiriya': (6.8765, 79.9891),
            'pannipitiya': (6.8449, 79.9607),
            
            # Western Province
            'wattala': (6.9890, 79.8917),
            'ja ela': (7.0747, 79.8910),
            'ja-ela': (7.0747, 79.8910),
            'kiribathgoda': (6.9804, 79.9297),
            'kelaniya': (6.9553, 79.9192),
            'gampaha': (7.0873, 80.0014),
            'kalutara': (6.5854, 79.9607),
            'panadura': (6.7132, 79.9026),
            'beruwala': (6.4788, 79.9827),
            'wadduwa': (6.6633, 79.9297),
            'horana': (6.7158, 80.0626),
            'matugama': (6.4896, 80.1628),
            'avissawella': (6.9522, 80.2095),
            'minuwangoda': (7.1727, 79.9533),
            'divulapitiya': (7.2232, 80.0078),
            'veyangoda': (7.1583, 80.0577),
            'nittambuwa': (7.1393, 80.0931),
            
            # Other Cities and Towns
            'matale': (7.4675, 80.6234),
            'dambulla': (7.8742, 80.6511),
            'chilaw': (7.5759, 79.7953),
            'kalmunai': (7.4088, 81.8356),
            'wattegama': (7.2869, 80.7020),
            'balangoda': (6.6521, 80.6997),
            'embilipitiya': (6.3429, 80.8502),
            'tangalle': (6.0241, 80.7959),
            'ambalantota': (6.1210, 81.0217),
            'deniyaya': (6.3442, 80.5544),
            'tissamaharama': (6.2843, 81.2874),
            'haputale': (6.7678, 80.9563),
            'bandarawela': (6.8318, 80.9854),
            'wellawaya': (6.7347, 81.1018),
        }
        
    def get_coordinates(self, location_name: str) -> tuple:
        """Convert location name to coordinates"""
        # Normalize the input
        normalized_name = location_name.lower().strip()
        
        # Direct match
        if normalized_name in self.location_database:
            coords = self.location_database[normalized_name]
            print(f"📍 Found exact match: {location_name} -> {coords}")
            return coords
        
        # Try partial matching
        for loc_key, coords in self.location_database.items():
            if normalized_name in loc_key or loc_key in normalized_name:
                print(f"📍 Found partial match: {location_name} -> {loc_key} -> {coords}")
                return coords
        
        # Default to Colombo if not found
        print(f"⚠️ Location '{location_name}' not found, defaulting to Colombo")
        return self.location_database['colombo']
    
    def get_location_name(self, location_input: str) -> str:
        """Get normalized location name"""
        normalized_name = location_input.lower().strip()
        
        # Direct match
        if normalized_name in self.location_database:
            return normalized_name.title()
        
        # Try partial matching
        for loc_key in self.location_database.keys():
            if normalized_name in loc_key or loc_key in normalized_name:
                return loc_key.title()
        
        return "Colombo"  # Default

# Your EXACT AI Service Classifier from notebook
class AIServiceClassifier:
    """Advanced AI service classifier using semantic understanding"""

    def __init__(self):
        print("🧠 Initializing AI Service Classifier...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.label_encoder = LabelEncoder()
        self.trained = False

    def train(self, training_data):
        """Train using semantic embeddings and neural networks"""
        print("🎯 Training AI service classifier...")

        # Encode text to embeddings (semantic understanding)
        texts = training_data['problem_description'].tolist()
        labels = training_data['service_type'].tolist()

        print("📄 Generating semantic embeddings...")
        embeddings = self.sentence_model.encode(texts)

        # Encode labels
        encoded_labels = self.label_encoder.fit_transform(labels)

        # Train neural network on embeddings
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=1500,
            random_state=42
        )

        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, encoded_labels, test_size=0.2, random_state=42
        )

        print("📄 Training neural network...")
        self.classifier.fit(X_train, y_train)

        # Test accuracy
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ AI classifier trained!")
        print(f"🎯 Accuracy: {accuracy:.2%}")

        self.trained = True
        return accuracy

    def predict(self, text):
        """Predict service type using AI semantic understanding"""
        if not self.trained:
            raise Exception("AI model not trained yet!")

        # Generate semantic embedding
        embedding = self.sentence_model.encode([text])

        # Get probabilities from neural network
        probabilities = self.classifier.predict_proba(embedding)[0]

        # Get service names
        service_names = self.label_encoder.classes_

        # Sort by probability
        service_probs = list(zip(service_names, probabilities))
        service_probs.sort(key=lambda x: x[1], reverse=True)

        return service_probs[:3]  # Top 3 predictions

# Your EXACT AI Location Extractor from notebook
class AILocationExtractor:
    """AI-based location extraction using semantic similarity"""

    def __init__(self):
        print("📍 Initializing AI Location Extractor...")
        self.location_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.trained = False

    def train_location_model(self):
        """Train location model with Sri Lankan geographic data"""
        print("🗺️ Training AI location model...")

        # Comprehensive Sri Lankan location data
        self.location_data = {
            'colombo': (6.9271, 79.8612),
            'kandy': (7.2906, 80.6337),
            'galle': (6.0535, 80.2210),
            'negombo': (7.2084, 79.8380),
            'jaffna': (9.6615, 80.0255),
            'kurunegala': (7.4818, 80.3609),
            'anuradhapura': (8.3114, 80.4037),
            'matara': (5.9549, 80.5550),
            'ratnapura': (6.6828, 80.3992),
            'koswatta': (6.8875, 79.8800),
            'dehiwala': (6.8560, 79.8638),
            'mount lavinia': (6.8374, 79.8634),
            'moratuwa': (6.7730, 79.8816),
            'kotte': (6.8905, 79.9015),
            'nugegoda': (6.8649, 79.8997),
            'maharagama': (6.8482, 79.9298),
            'rajagiriya': (6.9084, 79.8916),
            'battaramulla': (6.8992, 79.9186),
            'malabe': (6.9147, 79.9731),
            'gampaha': (7.0873, 80.0014),
            'kalutara': (6.5854, 79.9607),
            'panadura': (6.7132, 79.9026),
            'kelaniya': (6.9553, 79.9192),
            'kaduwela': (6.9381, 79.9897),
            'pelawatta': (6.8461, 79.9062),
            'thalawathugoda': (6.8738, 79.9750),
            'homagama': (6.8441, 80.0022),
            'wattala': (6.9890, 79.8917),
            'ja ela': (7.0747, 79.8910),
            'kiribathgoda': (6.9804, 79.9297),
            'kottawa': (6.8207, 79.9097)
        }

        # Generate semantic embeddings for location names
        location_names = list(self.location_data.keys())
        self.location_embeddings = self.location_model.encode(location_names)
        self.location_names = location_names

        self.trained = True
        print("✅ AI location model trained!")

    def extract_location(self, text):
        """Extract location using AI semantic similarity"""
        if not self.trained:
            self.train_location_model()

        # Generate semantic embedding for input text
        text_embedding = self.location_model.encode([text])

        # Calculate semantic similarities
        similarities = cosine_similarity(text_embedding, self.location_embeddings)[0]

        # Find best semantic match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]

        if best_similarity > 0.25:  # AI confidence threshold
            location_name = self.location_names[best_match_idx]
            coordinates = self.location_data[location_name]
            print(f"📍 AI detected location: {location_name.title()} (confidence: {best_similarity:.2f})")
            return coordinates, location_name
        else:
            print(f"📍 Location not clearly detected, using default (Colombo)")
            return (6.9271, 79.8612), "colombo"

# Your EXACT AI Time Extractor from notebook
class AITimeExtractor:
    """AI-based time requirement extraction"""

    def __init__(self):
        print("⏰ Initializing AI Time Extractor...")
        self.time_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.trained = False

    def train_time_model(self):
        """Train AI time extraction model"""
        print("🕑 Training AI time extraction model...")

        # Comprehensive time expressions
        time_training_examples = {
            'immediate': [
                'urgent help needed right now', 'emergency repair immediately',
                'asap need help', 'critical situation now', 'emergency service needed',
                'urgent urgent urgent', 'help needed immediately', 'emergency call'
            ],
            'today': [
                'help needed today', 'service today please', 'can someone come today',
                'need help same day', 'today would be great', 'today if possible',
                'same day service needed', 'help needed this morning', 'this afternoon works'
            ],
            'tomorrow': [
                'tomorrow would work', 'help needed tomorrow', 'tomorrow morning preferred',
                'tomorrow afternoon fine', 'tomorrow evening works', 'next day service',
                'can wait until tomorrow', 'tomorrow is perfect', 'help tomorrow please'
            ],
            'this_week': [
                'sometime this week', 'within the week', 'this week would work',
                'monday works', 'tuesday is good', 'wednesday preferred',
                'thursday available', 'friday works', 'weekday is fine'
            ],
            'weekend': [
                'weekend would be better', 'saturday works', 'sunday is good',
                'weekend service needed', 'weekend availability', 'saturday or sunday',
                'weekend preferred', 'weekend works best', 'saturday morning'
            ],
            'flexible': [
                'whenever convenient', 'no particular rush', 'flexible with timing',
                'anytime works', 'not urgent', 'scheduled service fine',
                'whenever possible', 'no time pressure', 'flexible schedule'
            ]
        }

        # Create time training data
        time_training_data = []
        for time_type, examples in time_training_examples.items():
            for example in examples:
                time_training_data.append({
                    'text': example,
                    'time_type': time_type
                })

        time_df = pd.DataFrame(time_training_data)

        # Generate semantic embeddings
        texts = time_df['text'].tolist()
        labels = time_df['time_type'].tolist()

        embeddings = self.time_model.encode(texts)

        # Train AI classifier
        self.time_label_encoder = LabelEncoder()
        encoded_labels = self.time_label_encoder.fit_transform(labels)

        self.time_classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.time_classifier.fit(embeddings, encoded_labels)

        self.trained = True
        print("✅ AI time extraction model trained!")

    def extract_time_requirement(self, text):
        """Extract time requirement using AI semantic understanding"""
        if not self.trained:
            self.train_time_model()

        # Generate semantic embedding
        text_embedding = self.time_model.encode([text])

        # AI prediction
        prediction = self.time_classifier.predict(text_embedding)[0]
        probabilities = self.time_classifier.predict_proba(text_embedding)[0]

        # Decode label
        time_type = self.time_label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)

        print(f"⏰ AI detected time: {time_type} (confidence: {confidence:.2f})")
        return time_type, confidence

# Your EXACT Complete AI Recommendation System from notebook
class CompleteAIRecommendationSystem:
    """Complete AI-powered recommendation system"""

    def __init__(self):
        print("🤖 Initializing Complete AI Recommendation System...")
        self.service_classifier = AIServiceClassifier()
        self.location_extractor = AILocationExtractor()
        self.time_extractor = AITimeExtractor()
        self.trained = False

    def train_complete_system(self, training_data):
        """Train all AI components"""
        print("🚀 Training Complete AI System...")

        # Train service classifier
        service_accuracy = self.service_classifier.train(training_data)

        # Train location and time extractors
        self.location_extractor.train_location_model()
        self.time_extractor.train_time_model()

        self.trained = True
        print("✅ Complete AI system trained successfully!")
        return service_accuracy

    def get_ai_recommendations(self, query, workers_data, user_lat, user_lng, max_results=5):
        """Get AI-powered worker recommendations"""
        if not self.trained:
            raise Exception("AI system not trained yet!")

        print(f"📍 AI PROCESSING: '{query}'")
        print("="*60)

        # AI analysis
        service_predictions = self.service_classifier.predict(query)
        location_coords, location_name = self.location_extractor.extract_location(query)
        time_type, time_confidence = self.time_extractor.extract_time_requirement(query)

        print(f"🎯 AI Service Analysis: {[(s, f'{p:.1%}') for s, p in service_predictions]}")
        print(f"📍 AI Location: {location_name.title()} {location_coords}")
        print(f"⏰ AI Time: {time_type} (confidence: {time_confidence:.2f})")
        print()

        # Use user's actual coordinates
        user_coords = (user_lat, user_lng)

        # Convert to DataFrame if needed
        if isinstance(workers_data, dict):
            workers_df = pd.DataFrame(workers_data['workers'])
        else:
            workers_df = workers_data

        # Filter workers using AI predictions
        likely_services = [s for s, p in service_predictions if p > 0.1]
        relevant_workers = workers_df[workers_df['service_type'].isin(likely_services)].copy()

        if len(relevant_workers) == 0:
            likely_services = [service_predictions[0][0]]
            relevant_workers = workers_df[workers_df['service_type'].isin(likely_services)].copy()

        if len(relevant_workers) == 0:
            print("⚠ No suitable workers found")
            return [], {"error": "No suitable workers found"}

        # AI-powered scoring
        scored_workers = []
        for idx, worker in relevant_workers.iterrows():
            # Service match score (AI confidence)
            service_confidence = next((p for s, p in service_predictions if s == worker['service_type']), 0)
            service_score = service_confidence * 70  # 70 points max

            # Distance score (geographic proximity)
            worker_coords = (worker['location']['latitude'], worker['location']['longitude'])
            distance = geodesic(user_coords, worker_coords).kilometers
            distance_score = max(0, 20 - (distance * 0.2))  # 20 points max

            # Quality score (rating & experience)
            quality_score = (worker['rating'] / 5.0) * 10  # 10 points max

            total_score = service_score + distance_score + quality_score

            scored_workers.append({
                'worker': worker,
                'score': total_score,
                'distance_km': distance,
                'service_confidence': service_confidence
            })

        # Sort by AI score
        scored_workers.sort(key=lambda x: x['score'], reverse=True)

        return scored_workers[:max_results], {
            'service_predictions': [(s, f'{p:.1%}') for s, p in service_predictions],
            'detected_location': location_name,
            'time_requirement': time_type,
            'user_location': user_coords
        }

def create_ai_training_data():
    """Create comprehensive training data for AI learning (EXACT from your notebook)"""
    print("🎯 Creating AI training data with semantic understanding...")

    # Real-world problem descriptions with context (EXACT from your notebook)
    service_training_examples = {
        'plumbing': [
            'water is dripping from my bathroom faucet constantly',
            'toilet keeps overflowing and making a huge mess in bathroom',
            'kitchen sink drain is completely blocked and water backing up',
            'shower has no water pressure at all very frustrating',
            'water heater stopped working no hot water for days',
            'pipe burst in basement flooding the entire area',
            'bathroom sink leaking underneath causing water damage',
            'toilet not flushing properly waste backing up',
            'water pressure extremely low throughout entire house',
            'drain making gurgling sounds and smells terrible',
            'faucet handle broken cannot turn water on or off',
            'water bill very high suspect underground leak'
        ],
        'electrical': [
            'lights keep flickering throughout the entire house',
            'power outlet stopped working completely no electricity',
            'circuit breaker keeps tripping shutting off power repeatedly',
            'electrical switch making loud sparking sounds very dangerous',
            'half the house has no electricity sudden power loss',
            'ceiling fan stopped working motor making grinding noise',
            'electrical wiring looks old and potentially hazardous',
            'light fixtures need professional installation help',
            'power surges damaging electronic devices frequently',
            'electrical panel needs upgrade current one very old',
            'outlets not working in kitchen cannot use appliances',
            'electrical safety inspection needed for insurance'
        ],
        'appliance_repair': [
            'washing machine not spinning clothes staying soaking wet',
            'dryer making extremely loud rattling noises when running',
            'dishwasher not cleaning dishes properly leaves residue',
            'refrigerator not keeping food cold temperature rising',
            'microwave stopped heating food completely broken',
            'oven temperature not working correctly burning food',
            'washing machine leaking water all over laundry room',
            'appliance technician needed for equipment repair service',
            'home appliances malfunctioning need professional diagnosis',
            'kitchen equipment not functioning properly needs fixing',
            'laundry machines broken down need immediate repair',
            'appliance maintenance service required for warranty'
        ],
        'ac_repair': [
            'air conditioner stopped cooling room getting very hot',
            'ac unit making strange grinding mechanical noises',
            'aircon leaking water onto floor creating puddles',
            'central air system not working properly uneven cooling',
            'air conditioning blowing warm air instead of cold',
            'ac remote control not responding need manual repair',
            'hvac system needs professional maintenance check',
            'cooling system completely broken during hot weather',
            'air conditioner filter needs replacement service',
            'central cooling not reaching upstairs rooms properly',
            'ac compressor making loud noises need diagnosis',
            'air conditioning installation needed for new room'
        ],
        'general_maintenance': [
            'house needs comprehensive maintenance check multiple issues',
            'various small repairs needed around home property',
            'property upkeep and general maintenance service required',
            'handyman needed for multiple different household fixes',
            'general repair work needed throughout entire building',
            'home maintenance service required for aging property',
            'building needs comprehensive repairs before inspection',
            'various household problems need professional fixing',
            'property maintenance contract needed for ongoing care',
            'general handyman work required for multiple rooms',
            'house maintenance checklist items need completion',
            'property care and upkeep service needed regularly'
        ],
        'masonry': [
            'need to build a wall in my kitchen area',
            'construct new brick wall for room separation',
            'stone wall construction needed for garden area',
            'build retaining wall for landscaping project',
            'construct bathroom wall with proper materials',
            'need professional to build kitchen counter wall',
            'build outdoor wall for privacy and security',
            'concrete wall construction needed for basement',
            'brick wall repair and reconstruction required',
            'build decorative stone wall for entrance',
            'wall construction needed for home addition',
            'masonry work needed for structural wall building'
        ],
        'painting': [
            'living room walls need fresh paint job badly',
            'exterior house paint peeling off needs refinishing',
            'bedroom walls need complete color change makeover',
            'ceiling paint cracking and falling need repair',
            'interior painting needed throughout entire house',
            'wall surface preparation and professional painting required',
            'complete house painting project needed inside and out',
            'professional painter needed for multiple rooms',
            'paint job required for home renovation project',
            'wall refinishing needed due to water damage',
            'exterior painting needed before selling house',
            'interior design painting project professional help'
        ],
        'cleaning': [
            'house needs thorough deep cleaning service badly',
            'post construction cleanup required extensive mess',
            'office space needs professional cleaning service',
            'carpet cleaning and stain removal service needed',
            'window cleaning service needed building wide',
            'bathroom deep cleaning required professional help',
            'kitchen cleaning and sanitization service needed',
            'move in cleaning service needed new house',
            'spring cleaning service needed entire property',
            'commercial cleaning service required for office',
            'post party cleanup service needed immediately',
            'house cleaning service needed regular maintenance'
        ],
        'carpentry': [
            'kitchen cabinet door broken hanging off hinges',
            'wooden table leg wobbly unstable needs repair',
            'custom shelving installation needed for storage',
            'furniture repair and restoration professional service',
            'door frame damaged needs fixing properly',
            'wooden stairs making loud creaking sounds',
            'cabinet making and installation service required',
            'furniture assembly service required professional help',
            'wood work needed for home renovation project',
            'custom carpentry work needed for built ins',
            'furniture refinishing service needed antique pieces',
            'wooden deck repair needed professional assessment'
        ],
        'gardening': [
            'lawn grass overgrown and messy needs cutting',
            'garden plants need professional care and maintenance',
            'tree branches need trimming professional service',
            'landscape design and maintenance service required',
            'lawn mowing service required regular maintenance',
            'garden cleanup needed after storm damage',
            'plant care and fertilization service needed',
            'outdoor space needs professional landscaping help',
            'yard maintenance service needed overgrown property',
            'garden design consultation needed professional advice',
            'lawn care service needed regular weekly maintenance',
            'landscaping project needed complete yard makeover'
        ],
        'roofing': [
            'roof leaking badly during rain need repair',
            'roof tiles broken and missing after storm',
            'roof maintenance and inspection service needed',
            'gutter cleaning and repair service required',
            'roof replacement needed old tiles damaged',
            'roofing contractor needed for new installation',
            'roof waterproofing service needed urgently',
            'roof structure repair needed professional help',
            'roofing materials installation and replacement',
            'roof damage assessment needed after weather',
            'roof ventilation system needs professional work',
            'roofing project needed for home extension'
        ],
        'flooring': [
            'floor tiles broken and need replacement urgently',
            'wooden floor refinishing service needed badly',
            'floor installation needed for new room',
            'floor repair needed due to water damage',
            'flooring contractor needed for renovation project',
            'floor polishing and maintenance service required',
            'floor covering installation needed professional help',
            'floor leveling needed uneven surface',
            'floor restoration needed for old property',
            'flooring upgrade needed modern materials',
            'floor cleaning and sealing service required',
            'floor installation project needs professional work'
        ]
    }

    # Generate training dataset
    training_data = []
    for service_type, examples in service_training_examples.items():
        for example in examples:
            training_data.append({
                'problem_description': example,
                'service_type': service_type
            })

    training_df = pd.DataFrame(training_data)
    print(f"✅ Generated {len(training_df)} training examples")
    return training_df

# Global variables
ai_system = None
workers_database = None
location_converter = None

@app.on_event("startup")
async def startup_event():
    global ai_system, workers_database, location_converter
    
    # Initialize Location Converter
    location_converter = LocationConverter()
    print("✅ Location Converter initialized")
    
    # Initialize AI system
    ai_system = CompleteAIRecommendationSystem()
    training_data = create_ai_training_data()
    accuracy = ai_system.train_complete_system(training_data)
    
    # Load workers database
    try:
        with open('handyman_database_3000.json', 'r', encoding='utf-8') as f:
            workers_database = json.load(f)
        print(f"✅ Loaded {len(workers_database['workers'])} workers from database")
    except FileNotFoundError:
        print("⚠ Database file not found!")
        workers_database = None

@app.get("/")
async def root():
    return {
        "message": "Handyman AI Service API", 
        "status": "running", 
        "workers_count": len(workers_database['workers']) if workers_database else 0,
        "location_converter": "enabled"
    }

@app.post("/search", response_model=SearchResponse)
async def search_workers(request: SearchRequest):
    if not ai_system or not workers_database or not location_converter:
        raise HTTPException(status_code=500, detail="AI system not initialized")
    
    try:
        # Convert location name to coordinates
        user_lat, user_lng = location_converter.get_coordinates(request.location)
        location_name = location_converter.get_location_name(request.location)
        
        print(f"🌍 User location: {request.location} -> ({user_lat}, {user_lng})")
        
        # Get AI recommendations using your EXACT system
        recommendations, analysis = ai_system.get_ai_recommendations(
            request.description, 
            workers_database,
            user_lat, 
            user_lng
        )
        
        # Add the user's input location to analysis
        analysis['user_input_location'] = location_name
        
        # Convert to response format
        worker_responses = []
        for rec in recommendations:
            worker = rec['worker']
            worker_responses.append(WorkerResponse(
                worker_id=worker['worker_id'],
                worker_name=worker['worker_name'],
                service_type=worker['service_type'],
                rating=worker['rating'],
                experience_years=worker['experience_years'],
                daily_wage_lkr=worker['pricing']['daily_wage_lkr'],
                phone_number=worker['contact']['phone_number'],
                city=worker['location']['city'],
                distance_km=round(rec['distance_km'], 1),
                ai_confidence=round(rec['service_confidence'] * 100, 1),
                bio=worker['profile']['bio']
            ))
        
        return SearchResponse(workers=worker_responses, ai_analysis=analysis)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)