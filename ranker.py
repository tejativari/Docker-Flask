import os
import textract
import nltk
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify, make_response
from flask_restful import Resource, Api
from flask_cors import CORS
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords as stp
from nltk.corpus import wordnet

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

UPLOAD_FOLDER = r'./flask_file_copy'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'docx'])

app = Flask(__name__)

CORS(app, resources={r"/*":{"origin": "*"}})
app.config['CORS_HEADERS']='Content-Type'
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


lemmatizer = WordNetLemmatizer()
analyzer = TfidfVectorizer().build_analyzer()
def stemmed_words(doc):
    return (lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in analyzer(doc) if w not in set(stp.words('english')))


def get_tf_idf_cosine_similarity(compare_doc,doc_corpus):
    tf_idf_vect = TfidfVectorizer(analyzer=stemmed_words)
    tf_idf_req_vector = tf_idf_vect.fit_transform([compare_doc]).todense()
    tf_idf_resume_vector = tf_idf_vect.transform(doc_corpus).todense()
    tf_idf_req_vector = np.asarray(tf_idf_req_vector)
    tf_idf_resume_vector = np.asarray(tf_idf_resume_vector)
    cosine_similarity_list = []
    for i in range(len(tf_idf_resume_vector)):
        tf_idf_resume_vector_reshaped = np.reshape(tf_idf_resume_vector[i],(1,-1))
        cosine_similarity_output = cosine_similarity(tf_idf_req_vector,tf_idf_resume_vector_reshaped)
        cosine_similarity_list.append(cosine_similarity_output[0][0])
    return cosine_similarity_list

def get_content_as_string(filename):
    text = textract.process(filename)
    lower_case_string =  str(text.decode('utf-8')).lower()
    return lower_case_string


def process_files(req_doc_text,resume_docs,application_ids):
    resume_doc_text = []
    for idx, doct in enumerate(resume_docs):
        resume_doc_text.append(get_content_as_string(doct))
    cos_sim_list = get_tf_idf_cosine_similarity(req_doc_text,resume_doc_text)
    final_doc_rating_list = []
    zipped_docs = zip(cos_sim_list,resume_docs,application_ids)
    sorted_doc_list = sorted(zipped_docs, key = lambda x: x[0], reverse=True)
    for idx, element in enumerate(sorted_doc_list):
        doc_rating_list = {}
        doc_rating_list['application_id'] = ("{:}".format(element[2]))
        doc_rating_list['filename']=(os.path.basename(element[1]))
        doc_rating_list['percentage_match']=("{:.0%}".format(element[0]))
        doc_rating_list['rank'] = str(idx+1)
        final_doc_rating_list.append(doc_rating_list)
    return final_doc_rating_list

@app.route('/hello')
def hello():
   return 'Hello, the service is running.'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class check_for_file(Resource):
   def post (self):
         application_ids = request.form.get('application_ids')
         application_ids = application_ids.split(",")
         if 'resume_files' not in request.files:
            return make_response(jsonify({'message':'Select at least one resume File to proceed further'}),400)
         job_description = request.form.get('required_file')
         resume_files = request.files.getlist("resume_files")
         if resume_files == 0:
               return make_response(jsonify({'message':'Select atleast one resume file to proceed further'}),400)
         if (len(job_description)>0 and (len(resume_files) > 0)):
            abs_paths = []
            try:
               os.stat(app.config['UPLOAD_FOLDER'])
            except:
               os.mkdir(app.config['UPLOAD_FOLDER'])
            req_document = job_description
            for resumefile in resume_files:
               filename = resumefile.filename
               if allowed_file(filename):
                  abs_paths.append(UPLOAD_FOLDER + '\\' + filename)
                  try:
                     resumefile.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
                  except:
                     return make_response(jsonify({"message":"No Resume Files exist"}),400)
               else:
                  return make_response(jsonify({'message':'Allowed file types are txt, pdf, docx'}),400)
            try:
               result = process_files(req_document,abs_paths,application_ids)
            except Exception as e:
               return make_response(jsonify({"message":str(e)}),500)
            shutil.rmtree('./flask_file_copy')
            return make_response(jsonify({'message':'success','result':result}),200)

         else:
            return make_response(jsonify({'message':'Allowed file types are txt, pdf, docx'}),400)

api.add_resource(check_for_file,'/rr/fetchranks')

if __name__ == "__main__":
    app.run()