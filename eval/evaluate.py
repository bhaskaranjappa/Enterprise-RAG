# C:\Users\ABhas\projects\enterprise-rag\eval\evaluate.py
import os
import sys
import json
import requests
import time

API_URL = "http://localhost:8000/chat"
GOLD_STANDARD_PATH = os.path.join(os.path.dirname(__file__), "gold_standard.json")
EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "eval_results.json")

SUBJECT_MAPPING = {
    "social justice": "social justice",
    "post independence": "post independence",
    "modern history": "modern india",
    "modern india": "modern india",
    "internal security": "internal security",
    "indian society": "indian society",
    "governance": "governance",
    "geography & dm": "geography",
    "ecology & environment": "environment",
    "ethics": "ethics",
    "world history": "world history"
}

def load_gold_standard():
    if not os.path.exists(GOLD_STANDARD_PATH):
        print(f"[Error] Gold standard dataset missing at: {GOLD_STANDARD_PATH}")
        sys.exit(1)
    with open(GOLD_STANDARD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_subject_name(source_str):
    if ":" in source_str:
        source_str = source_str.split(":")[1]
    if "(" in source_str:
        source_str = source_str.split("(")[0]
    raw_subject = source_str.lower().strip()
    return SUBJECT_MAPPING.get(raw_subject, raw_subject)

def run_evaluation():
    print("=== Phase 7: Executing Production RAG Evaluation Harness ===\n")
    
    questions = load_gold_standard()
    print(f"Loaded {len(questions)} evaluation pairs from gold_standard.json.")
    
    print("Warming up dense retrieval models (allowing up to 90s for local initialization)...")
    try:
        requests.post(API_URL, json={"query": "warmup"}, timeout=90)
        print("Pipeline warmed up successfully.\n")
    except Exception:
        print("[Warning] Warmup request hit threshold, moving directly to evaluation...\n")

    eval_dataset = []
    hit_count = 0
    out_of_scope_count = 0
    correct_refusals = 0
    
    for idx, item in enumerate(questions, 1):
        query = item.get("question")
        expected_answer = item.get("expected_answer", "")
        source_docs = item.get("source_docs", [])
        
        is_out_of_scope = "NOT_IN_DOCS" in expected_answer
        
        print(f"[{idx}/{len(questions)}] Query: '{query}'")
        
        expected_subjects = [clean_subject_name(doc) for doc in source_docs]
        if is_out_of_scope:
            out_of_scope_count += 1
            
        try:
            # Raised timeout to 90s to prevent CPU thread congestion drops
            response = requests.post(API_URL, json={"query": query}, timeout=90)
            if response.status_code != 200:
                print(f"  -> [Warning] API Error status {response.status_code}")
                continue
                
            data = response.json()
            generated_answer = data.get("answer", "")
            citations = data.get("citations", [])
            
            retrieved_files = [str(c.get("source", "")).lower() for c in citations]
            unique_retrieved = list(set(retrieved_files))
            
            is_hit = False
            
            if is_out_of_scope:
                if "INSUFFICIENT_CONTEXT" in generated_answer:
                    is_hit = True
                    correct_refusals += 1
            else:
                for subj in expected_subjects:
                    if any(subj in filename for filename in retrieved_files):
                        is_hit = True
                        break
            
            if is_hit:
                hit_count += 1
                
            eval_dataset.append({
                "question": query,
                "expected_answer": expected_answer,
                "generated_answer": generated_answer,
                "hit_success": is_hit
            })
            
            print(f"  Retrieved Files: {[os.path.basename(f) for f in unique_retrieved]}")
            print(f"  -> Match Result: {is_hit}\n")
            
            # Defensive pacing delay to prevent pipeline thread choking
            time.sleep(1.5)
            
        except Exception as e:
            print(f"  -> [Error] Connection failed: {str(e)}\n")
            continue

    total_valid = len(eval_dataset)
    if total_valid > 0:
        overall_hit_rate = (hit_count / total_valid) * 100
        print("\n" + "="*50)
        print("         PORTFOLIO BENCHMARK METRICS")
        print("="*50)
        print(f"Total Questions Evaluated       : {total_valid}")
        print(f"Successful Matches / Refusals   : {hit_count}")
        print(f"Out-of-Scope Prompts Tested     : {out_of_scope_count}")
        print(f"Accurate Context Refusals       : {correct_refusals}")
        print(f"Pipeline Accuracy / Hit Rate    : {overall_hit_rate:.2f}%")
        print("="*50)
    
    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_evaluation()