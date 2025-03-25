from flask import Flask, request, render_template, jsonify
import os
import docx2txt
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ""
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        return ""

def extract_resume_details(text):
    """
    Extracts structured information (skills, experience, education) from resume text.
    """
    # Extract skills (basic example using keywords)
    skills = re.findall(r'\b(Python|JavaScript|Machine Learning|Flask|SQL|Data Science|TensorFlow|Pandas|Scikit-learn|Deep Learning)\b', text, re.I)

    # Extract education (looking for degrees and fields)
    education = re.findall(r'(Bachelor|Master|PhD)[^\.]+\b(Computer Science|Engineering|Data Science|Mathematics|Information Technology)\b', text, re.I)
    education = [" ".join(edu) for edu in education]

    # Extract experience (job titles & years)
    experience = re.findall(r'(Software Engineer|Data Scientist|ML Engineer|Developer)[^\.]+\d{4}', text, re.I)

    # Summary - first 3 lines as a basic summary
    summary = " ".join(text.split("\n")[:3])

    return {
        "full_text": text[:2000],  # Limit text to 2000 chars
        "summary": summary.strip() if summary.strip() else "No summary available",
        "skills": list(set(skills)),
        "education": education if education else ["Not available"],
        "experience": experience if experience else ["Not available"]
    }

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume_files = request.files.getlist('resumes')

        resumes = []
        filenames = []
        for resume_file in resume_files:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
            resume_file.save(filename)
            filenames.append(resume_file.filename)
            resumes.append(extract_text(filename))

        if not resumes or not job_description:
            return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

        # Vectorize job description and resumes
        vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
        vectors = vectorizer.toarray()

        # Calculate cosine similarities
        job_vector = vectors[0]
        resume_vectors = vectors[1:]
        similarities = cosine_similarity([job_vector], resume_vectors)[0]

        # Get top 5 matching resumes
        top_indices = similarities.argsort()[-5:][::-1]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(similarities[i] * 100, 2) for i in top_indices]  # Convert to percentage

        return render_template('matchresume.html', message="Top matching resumes:", top_resumes=top_resumes, similarity_scores=similarity_scores)

    return render_template('matchresume.html')

@app.route('/get_resume_text')
def get_resume_text():
    filename = request.args.get('filename')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(file_path):
        resume_text = extract_text(file_path)
        resume_details = extract_resume_details(resume_text)  # Extract structured details
        return jsonify(resume_details)
    else:
        return jsonify({"error": "Resume not found"}), 404

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
