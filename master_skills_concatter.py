import os
import numpy as np
import pandas as pd
# Directory where the skills.npy files are stored

df=pd.read_csv('TechData.csv')
# Find all skills.npy files
temp_skills=[]
all_skills = set()  # Use a set to store unique skills

# Load each skills.npy file and add skills to the set
# for skill_file in skill_files:
#     skills_path = os.path.join(INDEX_DIR, skill_file)
#     skills = np.load(skills_path, allow_pickle=True)
#     temp_skills.append(skills)
#     all_skills.update(skills)  # Add to set to ensure uniqueness

print(df)

for index,row in df.iterrows():
    all_skills.update(row.iloc[2])
# Convert set back to sorted list
unique_skills = sorted(all_skills)

# Save the final unique skills list
merged_skills_path ="./merged_unique_skills.npy"
np.save(merged_skills_path, unique_skills)
# print(len(temp_skills))
# df['skills']=temp_skills
# df.to_csv('categories_list.csv',index=False)
print(f"✅ Merged and saved {len(unique_skills)} unique skills to: {merged_skills_path}")
