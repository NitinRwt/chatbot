import os
import json
import gradio as gr
import google.generativeai as genai
from datetime import datetime
import time
from typing import List, Dict, Any, Tuple

# Import your existing modules
try:
    from intent_detector import detect_intent #type:ignore
    from rag_utils import retrieve_profiles, retrieve_jobs, get_stats
except ImportError:
    print("Warning: Could not import custom modules. Some features may not work.")
    def detect_intent(text, model=None):
        return "SEARCH"
    def retrieve_profiles(query, top_k=3):
        return []
    def retrieve_jobs(query, top_k=3):
        return []
    def get_stats():
        return {"profiles_loaded": 0, "jobs_loaded": 0}

# Configure Gemini
API_KEY = os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY) #type:ignore

# System prompt
SYSTEM_PROMPT = """
You are a professional AI assistant that can perform three main tasks:

1. **ONBOARD** - Help professionals create their profile (name, role, skills, experience, location)
2. **SEARCH** - Find relevant job opportunities based on user requirements
3. **POST** - Create structured job postings from client descriptions

Respond in a friendly, professional manner. Keep responses concise but informative.
"""

def get_intent_emoji(intent: str) -> str:
    """Get emoji for intent type."""
    emoji_map = {
        "ONBOARD": "👤",
        "SEARCH": "🔍", 
        "POST": "📝"
    }
    return emoji_map.get(intent, "🤖")

def format_response(response: str, intent: str) -> str:
    """Format response with intent indicator."""
    emoji = get_intent_emoji(intent)
    return f"{emoji} **{intent}**\n\n{response}"

def process_message(message: str, history: List[List[str]], model_name: str) -> Tuple[str, List[List[str]]]:
    """Process user message and return response with updated history."""
    if not message.strip():
        return "", history
    
    if not API_KEY:
        error_msg = "❌ **Error**: Please set your GOOGLE_API_KEY environment variable."
        history.append([message, error_msg])
        return "", history
    
    try:
        # Detect intent
        intent = detect_intent(message, model=model_name)
        
        # Get context based on intent
        context_block = ""
        if intent == "SEARCH":
            job_results = retrieve_jobs(message, top_k=3)
            if job_results:
                context_lines = []
                for idx, job in enumerate(job_results, 1):
                    context_lines.append(f"{idx}. {job['text'][:200]}...")
                context_block = f"\n\n**Available Jobs:**\n" + "\n".join(context_lines)
        elif intent == "ONBOARD":
            profile_results = retrieve_profiles(message, top_k=3)
            if profile_results:
                context_lines = []
                for idx, prof in enumerate(profile_results, 1):
                    context_lines.append(f"{idx}. {prof['text'][:200]}...")
                context_block = f"\n\n**Existing Profiles:**\n" + "\n".join(context_lines)
        
        # Prepare prompt
        full_prompt = SYSTEM_PROMPT + context_block + f"\n\nUser: {message}\nAssistant:"
        
        # Generate response
        model = genai.GenerativeModel(model_name)   #type:ignore
        response = model.generate_content(full_prompt)
        
        # Format response
        formatted_response = format_response(response.text, intent)
        
        # Update history
        history.append([message, formatted_response])
        
        return "", history
        
    except Exception as e:
        error_msg = f"❌ **Error**: {str(e)}"
        history.append([message, error_msg])
        return "", history

def get_system_stats() -> str:
    """Get system statistics."""
    try:
        stats = get_stats()
        return f"""
        📊 **System Status**
        - Profiles loaded: {stats['profiles_loaded']}
        - Jobs loaded: {stats['jobs_loaded']}
        - API Status: {'✅ Connected' if API_KEY else '❌ Not configured'}
        - Last updated: {datetime.now().strftime('%H:%M:%S')}
        """
    except:
        return "📊 **System Status**: Unable to fetch stats"

def create_sample_profiles() -> str:
    """Generate sample profile data for testing."""
    sample_data = [
        {
            "id": "prof_001",
            "name": "Alice Johnson",
            "role": "Full Stack Developer",
            "skills": ["React", "Node.js", "Python", "PostgreSQL"],
            "experience": "5 years",
            "location": "San Francisco, CA"
        },
        {
            "id": "prof_002", 
            "name": "Bob Smith",
            "role": "Data Scientist",
            "skills": ["Python", "Machine Learning", "TensorFlow", "SQL"],
            "experience": "3 years",
            "location": "New York, NY"
        }
    ]
    
    os.makedirs("data", exist_ok=True)
    with open("data/onboarding_profiles.jsonl", "w") as f:
        for profile in sample_data:
            f.write(json.dumps(profile) + "\n")
    
    return "✅ Sample profiles created successfully!"

def create_sample_jobs() -> str:
    """Generate sample job data for testing."""
    sample_data = [
        {
            "id": "job_001",
            "title": "Senior React Developer",
            "company": "TechCorp",
            "type": "Remote",
            "skills": ["React", "TypeScript", "Node.js"],
            "description": "Build scalable web applications using React and TypeScript",
            "salary": "$80k-$120k",
            "location": "Remote"
        },
        {
            "id": "job_002",
            "title": "Python Data Engineer", 
            "company": "DataFlow Inc",
            "type": "Hybrid",
            "skills": ["Python", "Apache Spark", "AWS"],
            "description": "Design and maintain data pipelines for machine learning models",
            "salary": "$90k-$130k",
            "location": "Seattle, WA"
        }
    ]
    
    os.makedirs("data", exist_ok=True)
    with open("data/job_listings.jsonl", "w") as f:
        for job in sample_data:
            f.write(json.dumps(job) + "\n")
    
    return "✅ Sample jobs created successfully!"

# Custom CSS for styling
custom_css = """
/* Main container styling */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Header styling */
.header-section {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.header-section h1 {
    color: white;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.header-section p {
    color: rgba(255, 255, 255, 0.9);
    text-align: center;
    font-size: 1.1em;
}

/* Chat interface styling */
.chat-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
}

/* Message styling */
.message {
    padding: 15px;
    margin: 10px 0;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.user-message {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white;
    margin-left: 20%;
}

.assistant-message {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
    color: #1e293b;
    margin-right: 20%;
}

/* Button styling */
.action-button {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 25px;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.action-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}

/* Input styling */
.input-box {
    border: 2px solid #e5e7eb;
    border-radius: 15px;
    padding: 15px;
    font-size: 16px;
    transition: border-color 0.3s ease;
}

.input-box:focus {
    border-color: #4f46e5;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
}

/* Stats panel styling */
.stats-panel {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 15px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: white;
}

/* Responsive design */
@media (max-width: 768px) {
    .gradio-container {
        padding: 10px;
    }
    
    .header-section h1 {
        font-size: 2em;
    }
    
    .user-message, .assistant-message {
        margin-left: 5%;
        margin-right: 5%;
    }
}
"""

# Sample conversation starters
sample_conversations = [
    "I'm a Python developer with 5 years experience in Django and React, based in San Francisco",
    "Show me remote React developer jobs paying over $80k",
    "I need to hire a full-stack developer for my startup, budget is $5000 for 3 months",
    "Find me data science positions in New York",
    "Looking for freelance web development work"
]

def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(css=custom_css, title="Multi-Task Gemini Assistant") as demo:
        # Header
        gr.HTML("""
        <div class="header-section">
            <h1>🚀 Multi-Task Gemini Assistant</h1>
            <p>Your AI-powered career companion for onboarding, job search, and posting opportunities</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    height=500,
                    label="💬 Chat with Gemini",
                    elem_classes=["chat-container"],
                    bubble_full_width=False,
                    show_label=True,
                    avatar_images=("👤", "🤖")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message here... (e.g., 'I'm a Python developer looking for work')",
                        label="Your Message",
                        elem_classes=["input-box"],
                        scale=4
                    )
                    send_btn = gr.Button("Send 🚀", elem_classes=["action-button"], scale=1)
                
                # Sample conversation starters
                gr.Markdown("### 💡 Try these examples:")
                sample_buttons = []
                for i, sample in enumerate(sample_conversations):
                    btn = gr.Button(f"💬 {sample[:50]}{'...' if len(sample) > 50 else ''}", 
                                  size="sm", variant="secondary")
                    sample_buttons.append(btn)
            
            with gr.Column(scale=1):
                # Control panel
                gr.Markdown("### ⚙️ Settings")
                
                model_dropdown = gr.Dropdown(
                    choices=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
                    value="gemini-1.5-flash",
                    label="🧠 Model",
                    info="Choose your Gemini model"
                )
                
                # System stats
                stats_display = gr.Markdown(
                    get_system_stats(),
                    label="📊 System Status",
                    elem_classes=["stats-panel"]
                )
                
                refresh_stats_btn = gr.Button("🔄 Refresh Stats", size="sm")
                
                gr.Markdown("### 🛠️ Quick Actions")
                
                create_profiles_btn = gr.Button("👥 Create Sample Profiles", variant="secondary")
                create_jobs_btn = gr.Button("💼 Create Sample Jobs", variant="secondary")
                clear_chat_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                
                # Status messages
                status_msg = gr.Markdown("", elem_classes=["stats-panel"])
        
        # Event handlers
        def send_message(message, history, model):
            return process_message(message, history, model)
        
        def set_sample_message(sample_text):
            return sample_text
        
        def clear_chat():
            return []
        
        def refresh_stats():
            return get_system_stats()
        
        # Bind events
        send_btn.click(
            send_message,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            send_message,
            inputs=[msg, chatbot, model_dropdown],
            outputs=[msg, chatbot]
        )
        
        # Sample button events
        for i, btn in enumerate(sample_buttons):
            btn.click(
                lambda sample=sample_conversations[i]: sample,
                outputs=msg
            )
        
        refresh_stats_btn.click(refresh_stats, outputs=stats_display)
        clear_chat_btn.click(clear_chat, outputs=chatbot)
        create_profiles_btn.click(create_sample_profiles, outputs=status_msg)
        create_jobs_btn.click(create_sample_jobs, outputs=status_msg)
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: rgba(255,255,255,0.8);">
            <p>🌟 Powered by Google's Gemini AI | Built with ❤️ using Gradio</p>
            <p>💡 <strong>Tips:</strong> Use natural language to describe your needs. The AI will automatically detect if you want to onboard, search for jobs, or post a job.</p>
        </div>
        """)
    
    return demo

def main():
    """Main function to launch the interface."""
    print("🚀 Starting Multi-Task Gemini Assistant...")
    
    # Check API key
    if not API_KEY:
        print("⚠️  Warning: GOOGLE_API_KEY not found. Please set it as an environment variable.")
        print("   Example: export GOOGLE_API_KEY='your-api-key-here'")
    
    # Create and launch interface
    demo = create_interface()
    
    # Launch with custom settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True,        # Show errors in interface
        inbrowser=True,         # Open in browser automatically
        favicon_path=None,      # Add custom favicon if needed
        ssl_verify=False        # Disable SSL verification for development
    )

if __name__ == "__main__":
    main()