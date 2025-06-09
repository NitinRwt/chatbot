import os
import google.generativeai as genai

# Configure Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Error: Set GOOGLE_API_KEY environment variable before running.")
genai.configure(api_key=API_KEY)

def detect_intent(user_message: str, model: str = "gemini-1.5-flash") -> str:
    """
    Detect the intent of a user message using Gemini.
    
    Args:
        user_message: The user's input message
        model: The Gemini model to use for intent detection
        
    Returns:
        One of: "ONBOARD", "SEARCH", "POST"
    """
    
    intent_prompt = f"""
    Analyze the following user message and classify it into exactly one of these three intents:

    1. ONBOARD - User wants to create/register their professional profile (e.g., "I'm a developer with 5 years experience", "Register me as a data scientist")
    
    2. SEARCH - User wants to find job opportunities (e.g., "Show me React jobs", "Find remote developer positions", "I'm looking for work")
    
    3. POST - User wants to create a job posting/listing (e.g., "I need to hire a developer", "Looking for someone to build a website", "Need a freelancer for my project")

    User message: "{user_message}"

    Respond with ONLY one word: ONBOARD, SEARCH, or POST
    """
    
    try:
        # Initialize the model
        gemini_model = genai.GenerativeModel(model)
        
        # Generate response
        response = gemini_model.generate_content(intent_prompt)
        intent = response.text.strip().upper()
        
        # Validate the response
        if intent in ["ONBOARD", "SEARCH", "POST"]:
            return intent
        else:
            # Fallback logic based on keywords if Gemini gives unexpected response
            return fallback_intent_detection(user_message)
            
    except Exception as e:
        print(f"[Warning] Intent detection API call failed: {e}")
        # Use fallback method
        return fallback_intent_detection(user_message)

def fallback_intent_detection(user_message: str) -> str:
    """
    Fallback intent detection using keyword matching.
    
    Args:
        user_message: The user's input message
        
    Returns:
        One of: "ONBOARD", "SEARCH", "POST"
    """
    message_lower = user_message.lower()
    
    # Keywords for each intent
    onboard_keywords = [
        "i'm a", "i am a", "my name is", "register", "onboard", "profile", 
        "my skills are", "experience", "years of experience", "based in",
        "located in", "developer with", "engineer with"
    ]
    
    search_keywords = [
        "find", "search", "looking for", "show me", "jobs", "opportunities", 
        "hiring", "remote", "work", "position", "role", "career", "employment",
        "job listings", "openings"
    ]
    
    post_keywords = [
        "need", "hire", "looking to hire", "want to hire", "project", 
        "freelancer", "contractor", "build", "develop", "create", "budget",
        "pay", "client", "business needs"
    ]
    
    # Count matches for each intent
    onboard_score = sum(1 for keyword in onboard_keywords if keyword in message_lower)
    search_score = sum(1 for keyword in search_keywords if keyword in message_lower)
    post_score = sum(1 for keyword in post_keywords if keyword in message_lower)
    
    # Return the intent with the highest score
    if onboard_score > search_score and onboard_score > post_score:
        return "ONBOARD"
    elif search_score > post_score:
        return "SEARCH"
    else:
        return "POST"

# Test function
def test_intent_detection():
    """Test the intent detection with sample messages."""
    test_cases = [
        ("I'm a Python developer with 5 years of experience", "ONBOARD"),
        ("Show me remote React jobs", "SEARCH"),
        ("I need to hire a web developer for my startup", "POST"),
        ("Looking for machine learning positions", "SEARCH"),
        ("My name is John and I'm a data scientist", "ONBOARD"),
        ("Need someone to build a mobile app, budget $5000", "POST")
    ]
    
    print("Testing intent detection...")
    for message, expected in test_cases:
        detected = detect_intent(message)
        status = "✓" if detected == expected else "✗"
        print(f"{status} '{message}' -> {detected} (expected: {expected})")

if __name__ == "__main__":
    test_intent_detection()