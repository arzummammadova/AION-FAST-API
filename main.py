# ai-service/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
import uvicorn
import os
import shutil
import re
from typing import List, Dict, Any, Optional
from docx import Document
from pypdf import PdfReader
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering 
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    logging.info("NLTK məlumatları uğurla yükləndi.")
except Exception as e:
    logging.error(f"NLTK məlumatları yüklənərkən xəta: {e}")

UPLOAD_DIR = "uploaded_files"
try:
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)
        logging.info(f"Yükləmə qovluğu '{UPLOAD_DIR}' yaradıldı.")
except OSError as e:
    logging.error(f"Yükləmə qovluğu yaradıla bilmədi: {e}")

app = FastAPI(
    title="AION-File Professional AI Analysis Service",
    description="Professional sənəd analizi və kontekst dərki xidməti.",
    version="3.0.0"
)

class EnhancedAnalysisResponse(BaseModel):
    status: str
    message: str
    document_info: Dict[str, Any]
    content_analysis: Dict[str, Any]
    ai_insights: Dict[str, Any]
    question_answer: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    metadata: Dict[str, Any]
    analyzed_text: str

class AIModels:
    def __init__(self):
        self.qa_pipeline = None
        self.summarizer = None
        self.sentiment_analyzer = None
        self.ner_pipeline = None
        self.sentence_model = None
        self.tokenizer = None
        self.nlp = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_models(self):
        try:
            logging.info(f"Professional AI modelləri {self.device} üzərində yüklənir...")
            
            logging.info("Question-Answering modeli yüklənir...")
            self.qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                device=0 if self.device == "cuda" else -1
            )
            
            logging.info("Summarization modeli yüklənir...")
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            
            logging.info("Sentiment Analysis modeli yüklənir...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            logging.info("NER modeli yüklənir...")
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            
            logging.info("Sentence Transformer modeli yüklənir...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
            
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("spaCy modeli yükləndi.")
            except Exception as e: 
                logging.warning(f"spaCy modeli yüklənərkən xəta: {e}. Bəzi xüsusiyyətlər məhdud olacaq. 'python -m spacy download en_core_web_sm' əmrini icra etdiyinizdən əmin olun.")
            
            logging.info("Bütün professional AI modelləri uğurla yükləndi!")
            
        except Exception as e:
            logging.error(f"Model yükləmə xətası: {e}")
            
ai_models = AIModels()

class ProfessionalTextProcessor:
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.,!?;:()\-\'\"\/]', '', text) 
        return text.strip()
    
    @staticmethod
    def split_into_chunks(text: str, max_length: int = 400, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk = ' '.join(words[i:i + max_length])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks

    @staticmethod
    def extract_document_structure(text: str) -> Dict[str, Any]:
        lines = text.split('\n')
        
        headings = []
        paragraphs = []
        lists = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if len(line) < 100 and (line.isupper() or line.istitle()):
                headings.append(line)
            elif re.match(r'^[\d\.\-\*\+]\s', line):
                lists.append(line)
            else:
                paragraphs.append(line)
        
        return {
            "total_lines": len(lines),
            "headings": headings[:10],
            "paragraph_count": len(paragraphs),
            "list_items": len(lists),
            "estimated_reading_time": len(text.split()) // 200
        }
    
    @staticmethod
    def analyze_content_topics(text: str, nlp_model=None) -> Dict[str, Any]:
        words = text.lower().split()
        
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}

        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        word_freq = Counter(filtered_words).most_common(20)
        
        categories = {
            'business': ['company', 'business', 'market', 'financial', 'revenue', 'profit', 'customer', 'client', 'service', 'product', 'management', 'strategy', 'investment', 'budget'],
            'technical': ['system', 'software', 'technology', 'development', 'programming', 'database', 'server', 'network', 'security', 'algorithm', 'code', 'technical', 'implementation'],
            'legal': ['contract', 'agreement', 'legal', 'law', 'regulation', 'compliance', 'liability', 'terms', 'conditions', 'rights', 'obligations', 'clause'],
            'academic': ['research', 'study', 'analysis', 'methodology', 'results', 'conclusion', 'hypothesis', 'data', 'findings', 'academic', 'scientific', 'theory'],
            'medical': ['patient', 'medical', 'health', 'treatment', 'diagnosis', 'symptoms', 'doctor', 'hospital', 'medication', 'therapy', 'clinical'],
            'education': ['student', 'education', 'learning', 'teaching', 'school', 'university', 'course', 'curriculum', 'assessment', 'knowledge'],
            'data_science_ml': ['algorithm', 'model', 'machine learning', 'data science', 'neural network', 'deep learning', 'prediction', 'classification', 'regression', 'dataset', 'training', 'validation', 'feature', 'hyperparameter', 'accuracy', 'precision', 'recall', 'f1-score', 'gradient', 'optimization', 'tensor', 'framework', 'library', 'python', 'r', 'scikit-learn', 'tensorflow', 'pytorch', 'keras', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'statistical', 'analysis', 'big data', 'artificial intelligence', 'ai', 'suni intellekt', 'maşın öyrənməsi', 'alqoritm', 'model', 'təlim', 'proqnoz', 'təsnifat', 'reqressiya', 'hiperparametr', 'dəqiqlik', 'statistik', 'analiz']
        }
        
        category_scores = {}
        text_lower = text.lower()
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        dominant_topic = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else "general"
        
        return {
            "word_frequency": word_freq[:10],
            "dominant_topic": dominant_topic,
            "topic_scores": category_scores,
            "complexity_indicators": {
                "avg_word_length": np.mean([len(word) for word in words]) if words else 0, 
                "unique_words_ratio": len(set(words)) / len(words) if words else 0,
                "sentence_count": len(re.findall(r'[.!?]+', text))
            }
        }
    
    @staticmethod
    def extract_key_information(text: str, ner_pipeline=None) -> Dict[str, Any]:
        info = {
            "dates": [],
            "numbers": [],
            "emails": [],
            "urls": [],
            "phone_numbers": [],
            "entities": []
        }
        
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{2,4}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{2,4}\b'
        ]
        
        for pattern in date_patterns:
            info["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        info["numbers"] = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)
        
        info["emails"] = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        
        info["urls"] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        info["phone_numbers"] = re.findall(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', text)
        
        if ner_pipeline:
            try:
                ner_text = text[:5000] 
                entities = ner_pipeline(ner_text)
                info["entities"] = [{"text": ent["word"], "label": ent["entity_group"], "confidence": ent["score"].item()} for ent in entities if ent["score"] > 0.8] 
            except Exception as e:
                logging.error(f"NER analizi xətası: {e}")
                pass
        
        return info

class SmartDocumentAnalyzer:
    
    def __init__(self, models: AIModels):
        self.models = models
        self.processor = ProfessionalTextProcessor()
    
    def comprehensive_analysis(self, text: str, filename: str = "") -> Dict[str, Any]:
        analysis_start = datetime.now()
        
        structure = self.processor.extract_document_structure(text)
        
        content_topics = self.processor.analyze_content_topics(text, self.models.nlp)
        
        key_info = self.processor.extract_key_information(text, self.models.ner_pipeline)
        
        sentiment_result = self.analyze_sentiment(text)
        
        readability = self.analyze_readability(text)
        
        ai_summary = self.generate_smart_summary(text)
        
        key_insights = self.extract_key_insights(text)
        
        analysis_time = (datetime.now() - analysis_start).total_seconds()
        
        return {
            "document_info": {
                "filename": filename,
                "word_count": len(text.split()),
                "character_count": len(text),
                "analysis_time": f"{analysis_time:.2f} saniyə",
                **structure
            },
            "content_analysis": {
                "topics": content_topics,
                "key_information": key_info,
                "sentiment": sentiment_result,
                "readability": readability
            },
            "ai_insights": {
                "summary": ai_summary,
                "key_insights": key_insights,
                "document_purpose": self.determine_document_purpose(text, content_topics),
                "target_audience": self.determine_target_audience(text, content_topics, readability)
            }
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        if not self.models.sentiment_analyzer:
            logging.warning("Sentiment analiz modeli mövcud deyil.")
            return {"overall": "neutral", "confidence": 0.0, "details": "Model mövcud deyil"}
        
        try:
            chunks = [text[i:i+500] for i in range(0, len(text), 500)][:5]
            
            sentiments = []
            for chunk in chunks:
                if chunk.strip():
                    result = self.models.sentiment_analyzer(chunk)
                    sentiments.append(result[0])
            
            if sentiments:
                pos_count = sum(1 for s in sentiments if s['label'].lower() in ['positive', 'pos'])
                neg_count = sum(1 for s in sentiments if s['label'].lower() in ['negative', 'neg'])
                neu_count = sum(1 for s in sentiments if s['label'].lower() in ['neutral', 'neu'])
                
                total = len(sentiments)
                if pos_count > neg_count and pos_count > neu_count:
                    overall = "positive"
                elif neg_count > pos_count and neg_count > neu_count:
                    overall = "negative"
                else:
                    overall = "neutral"

                confidence = np.mean([s['score'] for s in sentiments])
                
                return {
                    "overall": overall,
                    "confidence": float(confidence),
                    "distribution": {
                        "positive": pos_count,
                        "negative": neg_count,
                        "neutral": neu_count
                    }
                }
            
        except Exception as e:
            logging.error(f"Sentiment analizi xətası: {e}")
            return {"overall": "neutral", "confidence": 0.0, "error": str(e)}
        
        return {"overall": "neutral", "confidence": 0.0}
    
    def analyze_readability(self, text: str) -> Dict[str, Any]:
        try:
            flesch_score = flesch_reading_ease(text)
            grade_level = flesch_kincaid_grade(text)
            
            if flesch_score >= 90:
                difficulty = "Çox asan"
            elif flesch_score >= 80:
                difficulty = "Asan"
            elif flesch_score >= 70:
                difficulty = "Orta-asan"
            elif flesch_score >= 60:
                difficulty = "Orta"
            elif flesch_score >= 50:
                difficulty = "Orta-çətin"
            elif flesch_score >= 30:
                difficulty = "Çətin"
            else:
                difficulty = "Çox çətin"
            
            return {
                "flesch_reading_ease": float(flesch_score),
                "grade_level": float(grade_level),
                "difficulty": difficulty,
                "reading_time_minutes": len(text.split()) // 200
            }
        except Exception as e:
            logging.error(f"Oxunabilirlik analizi xətası: {e}")
            return {"error": "Oxunabilirlik analizi mümkün olmadı"}
    
    def generate_smart_summary(self, text: str) -> str:
        if not self.models.summarizer:
            logging.warning("Xülasə modeli mövcud deyil.")
            return "Xülasə modeli mövcud deyil."
        
        if len(text.strip()) < 100:
            return "Sənəd xülasə edilə biləcək qədər uzun deyil."
            
        try:
            if len(text) > 3000: 
                chunks = ProfessionalTextProcessor.split_into_chunks(text, max_length=1000, overlap=800) 
                
                if not chunks:
                    logging.warning("Xülasə üçün mətn hissələri yaradıla bilmədi. Ehtiyat olaraq mətnin başlanğıcı qaytarılır.")
                    return f"Xülasə üçün mətn hissələri yaradıla bilmədi. Sənədin başlanğıcı: {text[:200]}..."

                summaries = []
                
                for chunk in chunks[:4]:
                    if len(chunk.strip()) > 100:
                        try:
                            summary = self.models.summarizer(
                                chunk,
                                max_length=150,
                                min_length=50,
                                do_sample=False,
                                num_beams=4
                            )
                            summaries.append(summary[0]['summary_text'])
                        except Exception as e:
                            logging.error(f"Hissə xülasə edilərkən xəta: {e}")
                            continue 
                
                if not summaries:
                    logging.warning("Uzun sənəd üçün xülasə yaradıla bilmədi. Ehtiyat olaraq mətnin başlanğıcı qaytarılır.")
                    return f"Uzun sənəd üçün xülasə yaradıla bilmədi. Sənədin başlanğıcı: {text[:200]}..."
                return " ".join(summaries)
            else:
                summary = self.models.summarizer(
                    text,
                    max_length=200,
                    min_length=60,
                    do_sample=False,
                    num_beams=4
                )
                return summary[0]['summary_text']
                
        except Exception as e:
            logging.error(f"Xülasə yaratma xətası: {e}")
            return f"Xülasə yaratma xətası: {str(e)}. Sənədin başlanğıcı: {text[:200]}..."
    
    def extract_key_insights(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
        
        important_keywords = ['important', 'significant', 'key', 'main', 'primary', 'essential', 'critical', 'major', 'fundamental', 'crucial', 'therefore', 'consequently', 'in conclusion', 'summary']
        
        insights = []
        for sentence in sentences[:20]:
            sentence_lower = sentence.lower()
            score = sum(1 for keyword in important_keywords if keyword in sentence_lower)
            
            if score > 0 or len(sentence) > 100:
                insights.append(sentence)
        
        return insights[:5]
    
    def determine_document_purpose(self, text: str, content_analysis: Dict) -> str:
        dominant_topic = content_analysis.get("topics", {}).get("dominant_topic", "general")
        
        purpose_indicators = {
            "report": ["analysis", "findings", "results", "conclusion", "methodology", "data"],
            "proposal": ["propose", "suggest", "recommend", "plan", "strategy", "objective"],
            "contract": ["agreement", "terms", "conditions", "parties", "obligations", "rights"],
            "manual": ["instructions", "guide", "steps", "procedure", "how to", "tutorial"],
            "presentation": ["slide", "presentation", "overview", "summary", "highlights"],
            "academic": ["research", "study", "hypothesis", "methodology", "references", "bibliography"],
            "news_article": ["breaking news", "report", "incident", "event", "investigation"],
            "blog_post": ["my thoughts", "opinion", "guide", "tips", "personal experience"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for purpose, indicators in purpose_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                scores[purpose] = score
        
        if scores:
            likely_purpose = max(scores.items(), key=lambda x: x[1])[0]
            return f"{likely_purpose.title()} ({dominant_topic} sahəsi)"
        
        return f"Ümumi sənəd ({dominant_topic} sahəsi)"
    
    def determine_target_audience(self, text: str, content_analysis: Dict, readability: Dict) -> str:
        grade_level = readability.get("grade_level", 12)
        dominant_topic = content_analysis.get("topics", {}).get("dominant_topic", "general")
        
        if grade_level <= 8:
            audience_level = "Geniş auditoriya (orta məktəb səviyyəsi)"
        elif grade_level <= 12:
            audience_level = "Orta təhsilli auditoriya (lisey səviyyəsi)"
        elif grade_level <= 16:
            audience_level = "Ali təhsilli auditoriya (universitet səviyyəsi)"
        else:
            audience_level = "Ekspert auditoriya (mütəxəssislər)"
        
        return f"{audience_level} ({dominant_topic} sahəsində maraqlananlar)"

class SmartQASystem:
    
    def __init__(self, models: AIModels):
        self.models = models
        self.processor = ProfessionalTextProcessor()
    
    def find_best_context(self, question: str, text_chunks: List[str]) -> tuple:
        if not self.models.sentence_model or not text_chunks:
            logging.warning("find_best_context: Sentence model və ya mətn hissələri mövcud deyil.")
            return text_chunks[0] if text_chunks else "", 0.0
            
        try:
            question_embedding = self.models.sentence_model.encode([question])
            chunk_embeddings = self.models.sentence_model.encode(text_chunks)
            
            similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
            best_idx = np.argmax(similarities)
            
            return text_chunks[best_idx], float(similarities[best_idx])
        except Exception as e:
            logging.error(f"find_best_context xətası: {e}")
            return text_chunks[0] if text_chunks else "", 0.0
    
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
        if not self.models.qa_pipeline:
            logging.warning("Sual-cavab modeli aktiv deyil.")
            return {
                "answer": "Sual-cavab modeli aktiv deyil",
                "confidence": 0.0
            }
            
        try:
            if len(context) > 3000:
                context = context[:3000]
            
            result = self.models.qa_pipeline(
                question=question,
                context=context,
                max_answer_len=300
            )
            
            return {
                "answer": result["answer"],
                "confidence": float(result["score"])
            }
            
        except Exception as e:
            logging.error(f"Sual cavablandırma xətası: {e}")
            return {
                "answer": f"Cavab vermə zamanı xəta: {str(e)}",
                "confidence": 0.0
            }

document_analyzer = SmartDocumentAnalyzer(ai_models)
qa_system = SmartQASystem(ai_models) 

def extract_text_from_file(file_path: str, mime_type: str) -> str:
    text = ""
    
    try:
        if mime_type == "application/pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
                
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file_path)
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
                    
        elif mime_type.startswith("text/"):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            logging.warning(f"Dəstəklənməyən fayl tipi: {mime_type}")
            return f"Dəstəklənməyən fayl tipi: {mime_type}"
            
    except Exception as e:
        logging.error(f"Mətn çıxarma xətası: {e}")
        return f"Mətn çıxarma xətası: {str(e)}"
        
    cleaned_text = ProfessionalTextProcessor.clean_text(text)
    if not cleaned_text.strip():
        logging.warning("Fayldan mətn çıxarıla bilmədi və ya boşdur.")
        return "Fayldan mətn çıxarıla bilmədi"
    return cleaned_text

@app.on_event("startup")
async def startup_event():
    ai_models.load_models()

@app.get("/")
async def root():
    return {"message": "AION-File Professional AI Analysis Service aktivdir!"}

@app.post("/analyze_document", response_model=EnhancedAnalysisResponse)
async def analyze_document(
    file: UploadFile = File(...),
    question: str | None = Form(None)
):
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"Fayl '{file.filename}' '{file_location}' ünvanına saxlandı.")
        
        extracted_text = extract_text_from_file(file_location, file.content_type)
        
        if "xəta" in extracted_text.lower() or "dəstəklənməyən" in extracted_text.lower():
            logging.error(f"Fayl '{file.filename}' üçün mətn çıxarma xətası: {extracted_text}")
            raise HTTPException(status_code=400, detail=extracted_text)
        
        analysis_result = document_analyzer.comprehensive_analysis(extracted_text, file.filename)
        
        question_answer = None
        if question and question.strip():
            logging.info(f"Sual aşkarlandı: '{question}'")
            chunks = ProfessionalTextProcessor.split_into_chunks(extracted_text, max_length=512, overlap=100)
            
            best_context = ""
            context_relevance = 0.0
            if ai_models.sentence_model and chunks:
                best_context, context_relevance = qa_system.find_best_context(question, chunks)
                logging.info(f"Ən uyğun kontekst skoru: {context_relevance:.2f}")
            elif chunks:
                best_context = chunks[0]
                context_relevance = 0.0
                logging.warning("Sentence model mövcud deyil, ilk mətn hissəsi kontekst kimi istifadə olunur.")
            else:
                best_context = extracted_text[:3000]
                logging.warning("Mətn hissələri yaradıla bilmədi, bütün mətnin başlanğıcı kontekst kimi istifadə olunur.")

            if ai_models.qa_pipeline:
                try:
                    qa_result = ai_models.qa_pipeline(
                        question=question,
                        context=best_context,
                        max_answer_len=300
                    )
                    
                    question_answer = {
                        "question": question,
                        "answer": qa_result["answer"],
                        "confidence": float(qa_result["score"]),
                        "context_relevance": float(context_relevance),
                        "reasoning": f"Bu cavab sənədin ən uyğun hissəsindən ({context_relevance:.2f} oxşarlıq skoru) əldə edilib."
                    }
                    logging.info(f"Sual cavablandırıldı. Cavab: '{qa_result['answer']}', Etibarlılıq: {qa_result['score']:.2f}")
                except Exception as e:
                    logging.error(f"Sual-cavab pipeline xətası: {e}")
                    question_answer = {
                        "question": question,
                        "answer": f"Sual cavablandırıla bilmədi: {str(e)}",
                        "confidence": 0.0,
                        "context_relevance": float(context_relevance) 
                    }
            else:
                logging.warning("Sual-cavab modeli aktiv deyil.")
                question_answer = {
                    "question": question,
                    "answer": "Sual-cavab modeli aktiv deyil.",
                    "confidence": 0.0,
                    "context_relevance": float(context_relevance) 
                }
        
        recommendations = generate_recommendations(analysis_result, question_answer)
        
        logging.info(f"'{file.filename}' faylı uğurla analiz edildi.")
        return EnhancedAnalysisResponse(
            status="success",
            message=f"'{file.filename}' professional analiz edildi",
            document_info=analysis_result["document_info"],
            content_analysis=analysis_result["content_analysis"],
            ai_insights=analysis_result["ai_insights"],
            question_answer=question_answer,
            recommendations=recommendations,
            metadata={
                "analysis_timestamp": datetime.now().isoformat(),
                "model_versions": {
                    "qa_model": "distilbert-base-cased-distilled-squad",
                    "summarizer": "facebook/bart-large-cnn",
                    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "ner": "dbmdz/bert-large-cased-finetuned-conll03-english",
                    "sentence_transformer": "all-MiniLM-L6-v2"
                }
            },
            analyzed_text=extracted_text
        )
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.critical(f"Ümumi analiz xətası: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analiz xətası: {str(e)}")
    finally:
        if os.path.exists(file_location):
            try:
                os.remove(file_location)
                logging.info(f"Müvəqqəti fayl '{file_location}' silindi.")
            except OSError as e:
                logging.error(f"Müvəqqəti fayl '{file_location}' silinərkən xəta: {e}")

def generate_recommendations(analysis: Dict, qa_result: Optional[Dict]) -> List[str]:
    recommendations = []
    
    doc_info = analysis.get("document_info", {})
    if doc_info.get("word_count", 0) > 5000:
        recommendations.append("Bu sənəd çox uzundur. Daha yaxşı oxunabilirlik üçün bölümlərlə işləyin.")
    
    readability = analysis.get("content_analysis", {}).get("readability", {})
    if readability.get("grade_level", 0) > 15:
        recommendations.append("Sənəd çox mürəkkəbdir. Sadələşdirmə üçün qısa cümlələr istifadə edin.")
    
    sentiment = analysis.get("content_analysis", {}).get("sentiment", {})
    if sentiment.get("overall") == "negative":
        recommendations.append("Sənədin ümumi tonu mənfidir. Konstruktiv ifadələr əlavə etməyi düşünün.")
    
    if qa_result and qa_result.get("confidence", 0) < 0.3:
        recommendations.append("Suala cavab etibarlılığı aşağıdır. Daha spesifik suallar verməyi cəhd edin.")
    
    dominant_topic = analysis.get("content_analysis", {}).get("topics", {}).get("dominant_topic", "")
    if dominant_topic == "data_science_ml":
        recommendations.append("Bu sənəd Data Science/Maşın Öyrənməsi mövzusundadır. Daha dəqiq analiz üçün domenə spesifik AI modellərindən istifadə etməyi və ya mövcud modelləri bu sahə üzrə təkrar təlim etməyi düşünün.")
        recommendations.append("Texniki terminlərin və kod parçalarının düzgün tanınması üçün xüsusi NER modelləri tətbiq edin.")
    
    if dominant_topic == "technical":
        recommendations.append("Texniki sənəd kimi, terminlər üçün izahat sözlüyü əlavə etməyi düşünün.")
    elif dominant_topic == "business":
        recommendations.append("Biznes sənədi kimi, əsas göstəriciləri və nəticələri vurğulayın.")
    
    return recommendations[:5]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
