import fitz  # PyMuPDF
import re
import os
import numpy as np
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import pandas as pd
import json
import concurrent.futures
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from map_clean_skills import map_and_clean_skills

app = Flask(__name__)
CORS(app)


api_keys = [
        "AIzaSyDxBnQ-CKHnVDUj69XCmu2ouX9-OyCbD9s",
        "AIzaSyDLx0s0LI3R93PPlDhz1_7RoS5ILrszJKA",
        "AIzaSyBpbNk3R9br6mDmQeqNkBhFMHYI6PIMGp0",
        "AIzaSyBkxdM0Fv3uGk77bBDQ-EI7UFIZ4bIxxEQ",
        "AIzaSyA6elRmgHFFWhrw1tVghpd4eWRTY-eImR0",
        "AIzaSyARZdv6cWGnkL99PkAQ9ItlgN1U2J4R7Fw",
        "AIzaSyDskxxa9LftXDM6oSvUTLH-NBXHHpe__sg",
        "AIzaSyAhjg1mVl8csbzf9UMDFPY2w_M3G1uvt7E",
        "AIzaSyAmiV6pFpo6E38fQCvrdn_m1HY-D0BgQB8",
        "AIzaSyD3fwrbbo5c7qgdFPZrt8szJ0M2AqJbzlI",
        "AIzaSyDM7fCF_Yt5vV5Q1hwS4JAZPL-GNMABHpQ"
    ]


def extract_text_from_pdf(pdf_path):
    print(f"Extracting text from: {pdf_path}")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    print("Text extraction complete.")
    return text

def sanitize_filename(filename):
    print(f"Sanitizing filename: {filename}")
    filename = re.sub(r'[\/\\:?"*<>|]', '_', filename)
    filename = filename.replace("&", "and").replace(" ", "_")
    print(f"Sanitized filename: {filename}")
    return filename

def extract_skills_and_certificates_with_gemini(pdf_path):
    print(f"Extracting skills and certificates from: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    prompt = f'''
    Extract the following information from the provided resume:

    1. List of technical and soft skills mentioned.
    2. List of certificates or certifications achieved.
    3. List of Job/Internship experiences mentioned.

    Ensure the output is provided in a structured JSON format:

    {{
      "skills": ["Skill1", "Skill2"],
      "certificates": ["Certificate1", "Certificate2"],
      "experiences": ["Experience1", "Experience2"]
    }}
    '''
    

    for api_key in api_keys:
        try:
            print(f"Trying API key: {api_key}")
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=text + "\n\n" + prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    temperature=0.7,
                )
            )
            
            text = response.text.replace('```json', '').replace('\n', '').replace('```', '')
            json_object = json.loads(text)
            print("Successfully extracted JSON data.")
            return json_object
        except Exception as e:
            print(f"API key {api_key} failed: {e}")

    print("All API keys failed.")
    return {"error": "All API keys failed"}

@app.route('/upload', methods=['POST'])
def upload():
    print("Request received!")
    
    if 'file' not in request.files:
        print("No file found in request")
        return jsonify({"error": "Missing file"}), 400
    
    file = request.files['file']
    print(f"Received file: {file.filename}")
    
    filename = sanitize_filename(file.filename)
    filepath = os.path.join("uploads", filename)
    
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)
    print(f"File saved at: {filepath}")
    
    extracted_data = extract_skills_and_certificates_with_gemini(filepath)
    print("Extraction process completed.")
    
    return extracted_data

def load_faiss_index(index_file, mapping_file):
    print(f"Loading FAISS index from {index_file} and mapping from {mapping_file}")
    if not os.path.exists(index_file) or not os.path.exists(mapping_file):
        print("Index or mapping file not found.")
        return None, None
    try:
        index = faiss.read_index(index_file)
        index_to_category = np.load(mapping_file, allow_pickle=True)
        print("FAISS index loaded successfully.")
        return index, index_to_category
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None, None



def extract_experiences_and_certificates_with_gemini(data,job_role):
    user_certifications, user_experiences=data['certifcates'], data['Experience']
    prompt = f"""
    Given the job role: {job_role}, certifications and professional experiences provided
    determine whether each certification and experience is relevant to the job role. and place true or false.

    Input Data:
    Certifications: {user_certifications}
    Experiences: {user_experiences}

    Ensure the output does not contain any explainations or code.
    Ensure the output is in JSON format only:
    {{
      "certifications_relevance": {{"Certification1": true, "Certification2": false}},
      "experiences_relevance": {{"Experience1": false, "Experience2": false}}
    }}
    """

    api_keys = [
        "AIzaSyDxBnQ-CKHnVDUj69XCmu2ouX9-OyCbD9s",
        "AIzaSyDLx0s0LI3R93PPlDhz1_7RoS5ILrszJKA",
        "AIzaSyBpbNk3R9br6mDmQeqNkBhFMHYI6PIMGp0",
        "AIzaSyBkxdM0Fv3uGk77bBDQ-EI7UFIZ4bIxxEQ",
        "AIzaSyA6elRmgHFFWhrw1tVghpd4eWRTY-eImR0",
        "AIzaSyARZdv6cWGnkL99PkAQ9ItlgN1U2J4R7Fw",
        "AIzaSyDskxxa9LftXDM6oSvUTLH-NBXHHpe__sg",
        "AIzaSyAhjg1mVl8csbzf9UMDFPY2w_M3G1uvt7E",
        "AIzaSyAmiV6pFpo6E38fQCvrdn_m1HY-D0BgQB8",
        "AIzaSyD3fwrbbo5c7qgdFPZrt8szJ0M2AqJbzlI",
        "AIzaSyDM7fCF_Yt5vV5Q1hwS4JAZPL-GNMABHpQ"
    ]

    for api_key in api_keys:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=1000,
                    temperature=0.9,
                )
            )

            text = response.text.strip().replace("```json", "").replace("```", "")
            # print(text)
            relevance_data = json.loads(text)

            return {
                "certifications_relevance": relevance_data.get("certifications_relevance", {}),
                "experiences_relevance": relevance_data.get("experiences_relevance", {})
            }

        except Exception as e:
            print(f"API key {api_key} failed: {e}")

    return {"error": "All API keys failed"}



def results_generator(data,category,temp, selectedCategory):
    category_skills_dir = "./categories_skill_files/"+selectedCategory+"_skill_files"
    # print('loading skill scores json files ',datetime.now())
    # print()
    salaries = pd.read_csv('./index/'+selectedCategory+'.csv')

    with open('./index/'+selectedCategory+'_skill_files.json', 'r') as file:
        skill_scores = json.load(file)
    category_skills_file = os.path.join(category_skills_dir, f"{category}_skills.npy")
    if os.path.exists(category_skills_file):
        # print('loading category skill files ')
        # print(datetime.now())
        category_skills = np.load(category_skills_file, allow_pickle=True).tolist()
        user_skills = data["Skills"]
        # print('calling clean map skills ',datetime.now())
        # print()
        
        cleaned_skills=map_and_clean_skills(user_skills,category_skills)
        
        category_skills=cleaned_skills.get('cleaned_job_role_skills',[])
        user_skills=cleaned_skills.get("mappped_User_skills",[])
        
        matching_skills = [x for x in category_skills if x in user_skills]
        skill_match=round(100*(len(matching_skills)/len(category_skills)),2)
        
        category=category.replace('__', ' ')
        category_skill_scores = skill_scores.get(f"{category}_skills ", {})
        
        total, user_score = sum(category_skill_scores.values()), sum(category_skill_scores.get(skill, 0) for skill in user_skills)
        
        skill_score_category=round(100*(user_score/total),2) if total!=0 else 0
        
        missing_skills=[x for x in category_skills if x not in user_skills]
        # print('calling salary_data')
        # print(datetime.now())
        # print('='*50)
        temp_result = salary_data(category, salaries)
        if not temp_result:
            return jsonify({"message": "No data found in the csv file"})
        
        if len(temp_result) == 2:
            salary_range, role_title = temp_result
            exp = None  # Set a default value for exp
        else:
            salary_range, role_title, exp = temp_result
        if salary_range:
            salary_range = salary_range.split('-')
            lower, higher = int(salary_range[0].strip()), int(salary_range[1].strip())
        else:
            lower, higher = 0, 0
        salary_estimate = (user_score / total) * lower if total else 0, (user_score / total) * higher if total else 0
        base_pay=[lower,higher]
    
    
        return({"experiences_relevance":temp["experiences_relevance"],"certifications_relevance":temp["certifications_relevance"],"role_title":role_title,"exp":exp,"category": category,"base_pay":base_pay,"salary_estimate":salary_estimate, "matching_skills": matching_skills,"skill_match":skill_match, "user_skill_score":skill_score_category,"missing_skills":missing_skills})
    else:
        return jsonify({"message": "No data found in the csv file"})




def salary_data(role, salaries):
    print(role)
    role=role.replace('_', '')
    role=role.replace('__','')    
    role=role.replace(' ', '')
    for _, row in salaries.iterrows():
        data_role=row['Role']
        data_role=re.sub(r'[\/\\:?"*<>|]', '_', data_role)
        data_role=data_role.replace('&', 'and')
        data_role=data_role.replace(' ', '')
        data_role=data_role.replace('_', '')
        data_role=data_role.replace('__','')
        #print(data_role)
        if data_role == role:
            return (row['Range'], row["Role"],row['Exp'])

    #print(role, 'did not match with anyone')
    return None

@app.route('/search', methods=['POST'])
def search():
    print("Processing search request.")
    data = request.json
    if not data or "Skills" not in data:
        print("Invalid input: Expected a list of skills.")
        return jsonify({"error": "Invalid input. Expected a list of skills."}), 400
    
    user_skills = data["Skills"]
    print(f"User skills: {user_skills}")
    
    selectedCategory = data['selectedCategory']
    print(f"Selected category: {selectedCategory}")
    
    if not isinstance(user_skills, list):
        return jsonify({"error": "Skills should be a list."}), 400
    
    index_file = "./index/"+selectedCategory+"_skill_categories_hnsw.index"
    mapping_file = "./index/"+selectedCategory+"_category_mapping.npy"
    skill_vocab_file = "./index/"+selectedCategory+"_master_skills.npy"
    
    index, index_to_category = load_faiss_index(index_file, mapping_file)
    if index is None or index_to_category is None:
        return jsonify({"error": "Failed to load FAISS index"}), 500
    
    if not os.path.exists(skill_vocab_file):
        return jsonify({"error": "Skill vocabulary file not found"}), 500
    
    unique_skills = np.load(skill_vocab_file, allow_pickle=True).tolist()
    mlb = MultiLabelBinarizer(classes=unique_skills)
    query_vector = mlb.fit_transform([user_skills]).astype(np.float32)
    
    if np.all(query_vector == 0):
        return jsonify({"message": "No matching skills found"})
    
    faiss.normalize_L2(query_vector)
    k = min(6, index.ntotal)
    distances, indices = index.search(query_vector, k)

    results = []
    # print(data)
    category_list=[]
    for x in indices[0]:
        if x < len(index_to_category):
            category = index_to_category[x]
            category_list.append(category)
    # print('calling experience and certificates ',datetime.now())
    # print()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        temp = list(executor.map(extract_experiences_and_certificates_with_gemini,[data] * len(category_list),category_list))
    # print('calling results generator',datetime.now())
    # print()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(results_generator,[data] * len(category_list),category_list,temp, [selectedCategory]*len(category_list)))

    results.sort(key=lambda x: (x["user_skill_score"]+x['skill_match'])//2, reverse=True)

    if not results:
        return jsonify({"message": "No relevant categories found"})

    final = []
    for x in range(len(results)): 
        final.append({
            "category": results[x]["role_title"],
            "matching_skills": results[x]["matching_skills"],
            "salary_estimate": results[x]["salary_estimate"],
            "Skill Score":  results[x]["user_skill_score"],
            "certifications_relevance":results[x]["certifications_relevance"],
            "Experience": results[x]["experiences_relevance"],
            "Base pay": results[x]["base_pay"],
            "Skill Match": results[x]["skill_match"],
            "message": "SUCCESS",
            "missing_skills": results[x]["missing_skills"]
        })

    return jsonify(final)
if __name__ == '__main__':
    app.run(debug=False,  use_reloader=False,threaded=True)