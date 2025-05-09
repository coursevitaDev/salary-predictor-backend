import fitz  # PyMuPDF
import re
import os
import numpy as np
from google import genai
from google.genai import types

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def sanitize_filename(filename):
    filename = re.sub(r'[\/\\:?"*<>|]', '_', filename)
    filename = filename.replace("&", "and")
    filename = filename.replace(" ", "_")
    return filename

def extract_skills_and_certificates_with_gemini(pdf_path, client):
    text = extract_text_from_pdf(pdf_path)

    prompt = f'''
    Extract the following information from the provided resume:

    1. List of technical and soft skills mentioned.
    2. List of certificates or certifications achieved and their relevance to job role: "java developer", if yes simply put True else False.

    Ensure the output is provided in a structured JSON format:

    {{
      "skills": ["Skill1", "Skill2", "Skill3"],
      "certificates": {{"Certificate1":False, "Certificate2":True}}
    }}
    '''

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=text + "\n\n" + prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=1000,
            temperature=0.7,
        )
    )

    return response.text

def evaluate_certificates(certificates, relevant_keywords):
    relevant_count = sum(1 for cert in certificates if any(keyword.lower() in cert.lower() for keyword in relevant_keywords))
    return relevant_count

def main():
    pdf_path = input("Enter the path to the resume PDF: ").strip()
    job_related_skills = ["Python", "Machine Learning", "AWS", "Docker"]

    api_keys = ["AIzaSyDxBnQ-CKHnVDUj69XCmu2ouX9-OyCbD9s"]
    client = genai.Client(api_key=api_keys[0])

    extracted_data = extract_skills_and_certificates_with_gemini(pdf_path, client)

    print("Extracted Data:", extracted_data)

if __name__ == "__main__":
    main()
