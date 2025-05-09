import google.generativeai as genai
from google.generativeai import types
import numpy as np
import os
import pandas as pd
import re
import json


# user_skills = [
#     "Python",
#     "ReactJs",
#     "Mongodb",
#     "NextJs",
#     "Pytorch",
#     "Power BI",
#     "Data Science",
#     "Machine Learning",
#     "Natural Language Processing",
#     "Operating System",
#     "Data Structures",
#     "Full Stack Web Development",
#     "Leadership",
#     "Adaptability",
#     "Consistent",
#     "Teamwork"
# ]

# job_skills = [
#     "Data Analysis",
#     "Statistical Modeling",
#     "Fraud Detection",
#     "Risk Assessment",
#     "Data Mining",
#     "Data Visualization",
#     "SQL",
#     "R",
#     "SAS",
#     "Excel",
#     "Tableau"
# ]

def map_and_clean_skills(user_skills, job_skills):
    api_keys=["AIzaSyDxBnQ-CKHnVDUj69XCmu2ouX9-OyCbD9s","AIzaSyDLx0s0LI3R93PPlDhz1_7RoS5ILrszJKA", "AIzaSyBpbNk3R9br6mDmQeqNkBhFMHYI6PIMGp0","AIzaSyBkxdM0Fv3uGk77bBDQ-EI7UFIZ4bIxxEQ", "AIzaSyA6elRmgHFFWhrw1tVghpd4eWRTY-eImR0","AIzaSyARZdv6cWGnkL99PkAQ9ItlgN1U2J4R7Fw","AIzaSyDskxxa9LftXDM6oSvUTLH-NBXHHpe__sg","AIzaSyAhjg1mVl8csbzf9UMDFPY2w_M3G1uvt7E","AIzaSyAmiV6pFpo6E38fQCvrdn_m1HY-D0BgQB8","AIzaSyD3fwrbbo5c7qgdFPZrt8szJ0M2AqJbzlI","AIzaSyDM7fCF_Yt5vV5Q1hwS4JAZPL-GNMABHpQ"]
    print('map_and_clean_skills')
    
    for api_key in api_keys:
        try:
            prompt=f'''
            Given two lists: (1) User Skills {user_skills} and (2) Job Role Skills {job_skills}, 
            unify the skills by mapping specific tools to their broader skill categories and removing redundant skills. 
            If a user skill corresponds to a broader category in the job role skills 
            (e.g., 'Power BI' maps to 'Data Visualization'), ensure it is mapped to job role skill (replace the 'Power  BI' to 'Data Visualization'). Additionally,
            if multiple specific tools exist under the same category in the job role list
            (e.g., both 'Power BI' and 'Tableau' under 'Data Visualization') but the user already has one of them, 
            only remove the redundant tools from the job role list. 
            do not remove any skills from user skill list
            The output should be a refined job role skills list that avoids redundancy 
            and aligns specific skills under their broader categories.
            Ensure the output does not have the code or explainations.
            Ensure the output is provided in a structured JSON format:
            {{
                "mappped_User_skills":
                ["skill1","skill2","skill3"], 
                "cleaned_job_role_skills":
                ["skill1", "skill2", "skill3"]
            }}

            '''
            genai.configure(api_key=api_key)
            # client = Client(api_key=api_key)

            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            response = model.generate_content(
                contents=prompt,
             
        
            )
            # response = client.models.generate_content(
            # model="gemini-2.0-flash",
            # contents=prompt,
            # config=types.GenerateContentConfig(
            #     max_output_tokens=5000,  # Increased to ensure completion
            #     temperature=0.9,
            # ))
            text = response.text
            text = text.replace('```json', '')  # Store the result back
            text = text.replace('\n', '')
            # text = text.replace(' ', '')

            text = text.replace('```', '')
            #print(text)
            json_string = text
            json_object = json.loads(json_string)
            return json_object
        except Exception as e:
            print(f"API key {api_key} failed: {e} in map")

    return {"error": "All API keys failed"}
    
    


        
