import os
import sys
import json
import google.generativeai as genai

from intent_detector import detect_intent
from rag_utils import retrieve_profiles, retrieve_jobs

# 1. Configure Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Error: Set GOOGLE_API_KEY environment variable before running.")
genai.configure(api_key=API_KEY)  #type:ignore

# 2. System prompt with few-shot examples for each mode: ONBOARD, SEARCH, POST
SYSTEM_PROMPT = """
You are a single AI assistant that can perform any of the following three tasks based on the user's request:

1. ONBOARD – Onboard a work professional by capturing their profile details such as name, role, skills, experience, and location. If information is missing, ask for it. If user already exists (by matching query to stored profiles), tell them they are already onboarded.

2. SEARCH – Search for job opportunities. Given a user's query (e.g., "Remote React developer paying at least $60k"), retrieve relevant job listings and present them in a concise format (title, company, type, salary).

3. POST – Create a structured job opportunity (for clients). Given a client's unstructured description (e.g., "I need someone to build a website for $2000 in 3 weeks"), produce a JSON‐style job post with fields: title, description, skills, budget, timeline.

Respond according to these rules:
- First, identify which of the three tasks the user is requesting (ONBOARD, SEARCH, or POST).
- Then, perform that task. If any details are missing from the user's request, ask a clarifying question.
- Always keep answers concise (2–4 sentences) and in plain English, except for the POST task, where you output a valid JSON object without extra markdown.

Below are three examples (few‐shot) to demonstrate expected behavior:

---

Example 1 – ONBOARD:
User: "I'm a backend developer with 4 years of experience in Python and Django, based in New York."
Assistant: "ONBOARD

Sure! Let me gather your onboarding details:
• Name: [please tell me your full name]
• Role: Backend Developer
• Skills: Python, Django
• Experience: 4 years
• Location: New York
Please confirm your full name so I can complete your profile."

---

Example 2 – SEARCH:
User: "Show me remote React jobs paying at least $50k/year."
Assistant: "SEARCH

Here are three remote React developer roles paying ≥ $50k/year:
1) Senior React Developer at AcmeCorp (Remote, USD 70k–90k)
2) Frontend Engineer at BetaTech (Remote, USD 55k–65k)
3) Fullstack Engineer at GammaWorks (Hybrid, USD 60k–80k)
Let me know if you want more details or a narrower focus."

---

Example 3 – POST:
User: "I need a WordPress expert to build a blog site, budget is around $1000."
Assistant: "POST

{
  \"title\": \"WordPress Developer for Blog Site\",
  \"description\": \"Create a blog site with a custom theme, SEO optimization, and social sharing features.\",
  \"skills\": [\"WordPress\", \"PHP\", \"HTML\", \"CSS\"],
  \"budget\": \"$1000\",
  \"timeline\": \"3 weeks\"
}"

---

Now, when a new user message arrives, first identify the intent (ONBOARD, SEARCH, or POST), and then follow the instructions above to respond properly.
"""

def chat_loop(model_name="gemini-1.5-flash"):
    """
    Main chat loop that handles conversation with context injection based on intent.
    """
    print(f"=== Multi-Task Gemini Chatbot (Model: {model_name}) ===")
    print("Type your message below. Type 'exit' to quit.\n")

    # Initialize the model
    model = genai.GenerativeModel(model_name) #type:ignore
    
    # Start a chat session with the system prompt
    chat = model.start_chat(history=[])

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # 1. Detect intent (ONBOARD / SEARCH / POST)
            intent = detect_intent(user_input, model=model_name)
            print(f"[Debug] Detected intent: {intent}")

            # 2. Depending on intent, retrieve context
            context_block = None
            if intent == "SEARCH":
                # Use RAG to get top-3 job listings
                job_results = retrieve_jobs(user_input, top_k=3)
                # Format as a plain text block
                lines = []
                for idx, job in enumerate(job_results, start=1):
                    lines.append(f"{idx}) {job['text']}")
                block_text = "\n".join(lines)
                context_block = f"Relevant job listings:\n{block_text}"
            elif intent == "ONBOARD":
                # Use RAG to find if a similar profile already exists
                profile_results = retrieve_profiles(user_input, top_k=3)
                lines = []
                for idx, prof in enumerate(profile_results, start=1):
                    lines.append(f"{idx}) {prof['text']}")
                block_text = "\n".join(lines)
                context_block = f"Existing profiles that match your query:\n{block_text}"
            # else: if intent == "POST", we do not need RAG because POST is rule‐based

            # 3. Prepare the full prompt
            full_prompt = SYSTEM_PROMPT
            if context_block:
                full_prompt += f"\n\nContext Information:\n{context_block}"
                print(f"[Debug] Injected context block:\n{context_block}\n")
            
            full_prompt += f"\n\nUser: {user_input}\nAssistant:"

            # 4. Generate response
            response = model.generate_content(full_prompt)
            assistant_reply = response.text

            # 5. Print assistant's reply
            print(f"\nBot: {assistant_reply}\n")

        except Exception as e:
            print(f"[Error] Gemini API call failed: {e}")
            print("Please try again or check your API key and internet connection.\n")
            continue

def main():
    """Main function to start the chatbot."""
    chosen_model = sys.argv[1] if len(sys.argv) > 1 else "gemini-1.5-flash"
    
    # Validate model name
    available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    if chosen_model not in available_models:
        print(f"Warning: Model '{chosen_model}' may not be available.")
        print(f"Available models: {', '.join(available_models)}")
        print(f"Using default: gemini-1.5-flash\n")
        chosen_model = "gemini-1.5-flash"
    
    chat_loop(model_name=chosen_model)

if __name__ == "__main__":
    main()