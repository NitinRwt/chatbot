import os
import json
import google.generativeai as genai
from typing import List, Dict, Any
import time

# 1. Configure Gemini SDK
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in env before running.")

genai.configure(api_key=API_KEY) # type: ignore

# 2. File paths
DATA_DIR = "data"
PROFILE_IN = os.path.join(DATA_DIR, "onboarding_profiles.jsonl")
JOB_IN = os.path.join(DATA_DIR, "job_listings.jsonl")
PROFILE_OUT = os.path.join(DATA_DIR, "embeddings_profiles.jsonl")
JOB_OUT = os.path.join(DATA_DIR, "embeddings_jobs.jsonl")

def get_embedding(text: str, model: str = "models/text-embedding-004", task_type: str = "retrieval_document") -> List[float]:
    """
    Get embedding for a single text using Gemini API.
    
    Args:
        text: Text to embed
        model: Embedding model to use
        task_type: Task type for the embedding
    
    Returns:
        List of floats representing the embedding vector
    """
    try:
        # Use the updated API for google-generativeai >= 0.8.0
        response = genai.embed_content( # type: ignore
            model=model,
            content=text,
            task_type=task_type,
            title=None  # Optional title for the content
        )
        return response['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Retry once after a short delay
        time.sleep(1)
        try:
            response = genai.embed_content( # type: ignore
                model=model,
                content=text,
                task_type=task_type
            )
            return response['embedding']
        except Exception as e2:
            print(f"Retry failed: {e2}")
            raise e2

def create_profile_text(record: Dict[str, Any]) -> str:
    """Create a comprehensive text representation of a profile."""
    text_parts = []
    
    if record.get('name'):
        text_parts.append(f"Name: {record['name']}")
    
    if record.get('role'):
        text_parts.append(f"Role: {record['role']}")
    
    if record.get('skills'):
        skills = record['skills']
        if isinstance(skills, list):
            text_parts.append(f"Skills: {', '.join(skills)}")
        else:
            text_parts.append(f"Skills: {skills}")
    
    if record.get('experience'):
        text_parts.append(f"Experience: {record['experience']}")
    
    if record.get('location'):
        text_parts.append(f"Location: {record['location']}")
    
    return ". ".join(text_parts) + "."

def create_job_text(record: Dict[str, Any]) -> str:
    """Create a comprehensive text representation of a job listing."""
    text_parts = []
    
    if record.get('title'):
        text_parts.append(f"Title: {record['title']}")
    
    if record.get('company'):
        text_parts.append(f"Company: {record['company']}")
    
    if record.get('type'):
        text_parts.append(f"Type: {record['type']}")
    
    if record.get('skills'):
        skills = record['skills']
        if isinstance(skills, list):
            text_parts.append(f"Required Skills: {', '.join(skills)}")
        else:
            text_parts.append(f"Required Skills: {skills}")
    
    if record.get('description'):
        text_parts.append(f"Description: {record['description']}")
    
    if record.get('location'):
        text_parts.append(f"Location: {record['location']}")
    
    return ". ".join(text_parts) + "."

def embed_and_write(in_path: str, out_path: str, is_profile: bool = True, 
                   embed_model: str = "models/text-embedding-004"):
    """
    Read JSONL from in_path, generate embeddings, and write to out_path.
    
    Args:
        in_path: Input JSONL file path
        out_path: Output JSONL file path
        is_profile: Whether processing profiles (True) or jobs (False)
        embed_model: Embedding model to use
    """
    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        return
    
    processed_count = 0
    error_count = 0
    
    with open(in_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                record = json.loads(line.strip())
                doc_id = record.get("id", f"unknown_{line_num}")
                
                # Create text representation
                if is_profile:
                    text = create_profile_text(record)
                    task_type = "retrieval_document"
                else:
                    text = create_job_text(record)
                    task_type = "retrieval_document"
                
                # Get embedding
                embedding = get_embedding(text, embed_model, task_type)
                
                # Create output object
                out_obj = {
                    "id": doc_id,
                    "text": text,
                    "embedding": embedding,
                    "original_data": record  # Keep original data for reference
                }
                
                # Write to output file
                f_out.write(json.dumps(out_obj) + "\n")
                processed_count += 1
                
                print(f"✓ Embedded {('profile' if is_profile else 'job')} {doc_id} "
                      f"(line {line_num})")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"✗ JSON decode error on line {line_num}: {e}")
                continue
            except Exception as e:
                error_count += 1
                print(f"✗ Error processing line {line_num}: {e}")
                continue
    
    print(f"\nProcessed: {processed_count}, Errors: {error_count}")

def main():
    """Main function to process both profiles and jobs."""
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Starting embedding generation...")
    print(f"Using embedding model: models/text-embedding-004")
    
    # Process profiles
    if os.path.exists(PROFILE_IN):
        print(f"\n📋 Processing profiles from {PROFILE_IN}")
        embed_and_write(PROFILE_IN, PROFILE_OUT, is_profile=True)
        print(f"✓ Profile embeddings saved to {PROFILE_OUT}")
    else:
        print(f"⚠️  Profile file not found: {PROFILE_IN}")
    
    # Process job listings
    if os.path.exists(JOB_IN):
        print(f"\n💼 Processing jobs from {JOB_IN}")
        embed_and_write(JOB_IN, JOB_OUT, is_profile=False)
        print(f"✓ Job embeddings saved to {JOB_OUT}")
    else:
        print(f"⚠️  Job file not found: {JOB_IN}")
    
    print("\n🎉 Embedding generation completed!")

if __name__ == "__main__":
    main()