from flask import Flask, request, render_template, jsonify
import os
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load Sentence-BERT Model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

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

@app.route("/")
def matchresume():
    return render_template('matchresume.html')
@app.route('/get_resume_text')
def get_resume_text():
    filename = request.args.get('filename')
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if os.path.exists(file_path):
        resume_text = extract_text(file_path)
        return jsonify({
            "full_text": resume_text[:3000]  # Limit text for better display
        })
    else:
        return jsonify({"error": "Resume not found"}), 404

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

        # Compute embeddings using Sentence-BERT
        job_embedding = sbert_model.encode(job_description, convert_to_tensor=True)
        resume_embeddings = [sbert_model.encode(resume, convert_to_tensor=True) for resume in resumes]

        # Calculate cosine similarity
        similarities = [util.pytorch_cos_sim(job_embedding, resume_embedding).item() for resume_embedding in resume_embeddings]

        # Get top 5 matching resumes
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:5]
        top_resumes = [filenames[i] for i in top_indices]
        similarity_scores = [round(similarities[i] * 100, 2) for i in top_indices]

        return render_template('matchresume.html', message="Top matching resumes:", top_resumes=top_resumes, similarity_scores=similarity_scores)

    return render_template('matchresume.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
