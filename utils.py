import re
import datetime
from typing import Dict, Any, List

def clean_response(text: str) -> str:
    """Remove think tags and clean response"""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

def extract_information(user_input: str) -> Dict[str, Any]:
    """Extract user information and topics from input"""
    lower_input = user_input.lower()
    extracted = {
        "topics": [],
        "user_info": {},
        "intent": None
    }
    
    # Extract name
    name_patterns = [
        r"my name is (\w+)",
        r"i'm (\w+)",
        r"i am (\w+)",
        r"call me (\w+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, lower_input)
        if match:
            name = match.group(1).capitalize()
            extracted["user_info"]["name"] = name
            break
    
    # Extract preferences
    if "i like" in lower_input:
        like_match = re.search(r"i like (.+?)(?:\.|,|$)", lower_input)
        if like_match:
            preference = like_match.group(1).strip()
            extracted["topics"].append("preferences")
            extracted["user_info"]["preference"] = preference
    
    # Extract age
    age_match = re.search(r"i am (\d+) years old|i'm (\d+)|my age is (\d+)", lower_input)
    if age_match:
        age = next(g for g in age_match.groups() if g)
        extracted["user_info"]["age"] = age
    
    # Detect topics
    topic_keywords = {
        "weather": ["weather", "temperature", "rain", "sunny", "cloudy"],
        "technology": ["computer", "program", "code", "tech", "software"],
        "music": ["music", "song", "listen", "band", "artist"],
        "food": ["eat", "food", "hungry", "cook", "recipe"],
        "health": ["health", "exercise", "sleep", "tired", "sick"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in lower_input for keyword in keywords):
            extracted["topics"].append(topic)
    
    return extracted

def get_time_greeting() -> str:
    """Get appropriate greeting based on time of day"""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    else:
        return "Good evening"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    minutes = int(seconds / 60)
    if minutes == 0:
        return "less than a minute"
    elif minutes == 1:
        return "about a minute"
    else:
        return f"about {minutes} minutes"