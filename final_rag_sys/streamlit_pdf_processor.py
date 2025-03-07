import streamlit as st
import pandas as pd
import plotly.express as px
import time
import random
import io
import base64
from PIL import Image
import re
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download NLTK resources silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="PDF to Structured Learning",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
    }
    .card {
        border-radius: 5px;
        padding: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .step-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .step-number {
        background-color: #1E88E5;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
        float: left;
    }
    .step-content {
        margin-left: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

# Function to identify potential sections in the text
def identify_sections(text):
    # Common section patterns in academic/educational PDFs
    section_patterns = [
        r'\n\s*(\d+\.?\s+[A-Z][A-Za-z\s]+)\s*\n',  # Numbered sections like "1. Introduction"
        r'\n\s*([A-Z][A-Za-z\s]+:)\s*\n',          # Sections ending with colon like "Introduction:"
        r'\n\s*([A-Z][A-Z\s]+)\s*\n',              # ALL CAPS sections like "INTRODUCTION"
        r'\n\s*(Chapter \d+:?\s+[A-Za-z\s]+)\s*\n'  # Chapter headings
    ]
    
    sections = []
    for pattern in section_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            sections.append((match.group(1), match.start()))
    
    # Sort sections by their position in the text
    sections.sort(key=lambda x: x[1])
    
    return sections

# Modified NLTK tokenization function to avoid punkt_tab dependency
def custom_sent_tokenize(text):
    """
    A custom sentence tokenizer that doesn't rely on punkt_tab
    """
    # Simple rule-based sentence splitting
    # Split on periods, exclamation marks, or question marks followed by a space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return sentences

# Replace the extract_key_concepts function
def extract_key_concepts(text, num_concepts=5):
    # Use our custom tokenizer instead of NLTK's
    sentences = custom_sent_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # Count word frequencies (excluding stopwords)
    word_freq = {}
    for sentence in sentences:
        words = sentence.lower().split()
        for word in words:
            # Clean the word (remove punctuation)
            word = re.sub(r'[^\w\s]', '', word)
            if word and word not in stop_words and len(word) > 3:
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
    
    # Get the most frequent words as key concepts
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    key_concepts = [word for word, freq in sorted_words[:num_concepts]]
    
    return key_concepts

# Function to generate quiz questions from text
def generate_quiz_questions(text, num_questions=3):
    # Use our custom tokenizer
    sentences = custom_sent_tokenize(text)
    
    # Filter sentences that might be good for questions (longer, informative sentences)
    potential_question_sentences = [s for s in sentences if len(s.split()) > 10 and '?' not in s]
    
    # If we don't have enough sentences, just use what we have
    if len(potential_question_sentences) < num_questions:
        potential_question_sentences = sentences
    
    # Select random sentences to turn into questions
    selected_sentences = random.sample(potential_question_sentences, min(num_questions, len(potential_question_sentences)))
    
    questions = []
    for i, sentence in enumerate(selected_sentences):
        # Create a simple question by finding a key term and asking about it
        words = sentence.split()
        if len(words) > 5:
            # Find a potential keyword (not at the beginning, not a common word)
            stop_words = set(stopwords.words('english'))
            keywords = [w for w in words[2:] if w.lower() not in stop_words and len(w) > 4]
            
            if keywords:
                keyword = random.choice(keywords)
                # Replace the keyword with a blank in the sentence
                blank_sentence = sentence.replace(keyword, "________")
                
                # Create question
                question = {
                    "id": i + 1,
                    "question": f"Fill in the blank: {blank_sentence}",
                    "answer": keyword,
                    "context": sentence
                }
                questions.append(question)
    
    return questions

# Function to create a learning module from text
def create_learning_module(text, title):
    # Identify sections
    sections = identify_sections(text)
    
    # If no sections found, create a default one
    if not sections:
        sections = [("Main Content", 0)]
    
    # Extract content for each section
    section_contents = []
    for i in range(len(sections)):
        start_pos = sections[i][1]
        end_pos = sections[i+1][1] if i < len(sections) - 1 else len(text)
        
        section_title = sections[i][0]
        section_content = text[start_pos:end_pos].strip()
        
        # Skip very short sections
        if len(section_content) > 100:
            section_contents.append({
                "title": section_title,
                "content": section_content,
                "key_concepts": extract_key_concepts(section_content)
            })
    
    # Generate quiz questions
    quiz_questions = generate_quiz_questions(text)
    
    # Create the learning module
    learning_module = {
        "title": title,
        "sections": section_contents,
        "quiz_questions": quiz_questions,
        "estimated_reading_time": max(5, len(text) // 1000)  # Rough estimate: 1000 chars ~= 1 minute
    }
    
    return learning_module

# Function to display the learning module
def display_learning_module(module):
    st.markdown(f"<h1 class='main-header'>{module['title']}</h1>", unsafe_allow_html=True)
    
    # Module overview
    st.markdown("### Module Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Sections:** {len(module['sections'])}")
    
    with col2:
        st.markdown(f"**Reading Time:** {module['estimated_reading_time']} minutes")
    
    with col3:
        st.markdown(f"**Quiz Questions:** {len(module['quiz_questions'])}")
    
    # Display sections
    st.markdown("### Module Content")
    
    for i, section in enumerate(module['sections']):
        with st.expander(f"{section['title']}"):
            # Display a preview of the content (first 300 chars)
            preview = section['content'][:300] + "..." if len(section['content']) > 300 else section['content']
            st.markdown(preview)
            
            # Display key concepts
            st.markdown("**Key Concepts:**")
            concepts_html = " • ".join([f"<span style='background-color: #e3f2fd; padding: 3px 8px; border-radius: 10px; margin-right: 5px;'>{concept}</span>" for concept in section['key_concepts']])
            st.markdown(concepts_html, unsafe_allow_html=True)
            
            # Option to view full content
            if st.button(f"View Full Content", key=f"view_full_{i}"):
                st.markdown("---")
                st.markdown(section['content'])
    
    # Display quiz
    st.markdown("### Knowledge Check")
    st.markdown("Test your understanding of the material with these questions:")
    
    for i, question in enumerate(module['quiz_questions']):
        st.markdown(f"**Question {i+1}:** {question['question']}")
        
        # Create a unique key for each text input
        user_answer = st.text_input("Your answer:", key=f"quiz_answer_{i}")
        
        if user_answer:
            if user_answer.lower() == question['answer'].lower():
                st.success("Correct! Well done.")
            else:
                st.error(f"Not quite. The correct answer is: {question['answer']}")
                st.info(f"Context: {question['context']}")
        
        st.markdown("---")

# Main app
def main():
    st.markdown("<h1 class='main-header'>Intelligent LMS: PDF to Structured Learning</h1>", unsafe_allow_html=True)
    st.markdown(
        "Transform unstructured PDF documents into structured, interactive learning modules."
    )
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.sidebar.title("Document Processor")
    
    st.sidebar.markdown("### How It Works")
    st.sidebar.markdown("""
    1. Upload a PDF document
    2. Our AI analyzes the content
    3. Content is structured into learning modules
    4. Key concepts are identified
    5. Interactive elements are generated
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This tool demonstrates how unstructured data can be transformed into actionable knowledge "
        "for educational purposes. Upload any PDF to see it converted into an interactive learning module."
    )
    
    # Main content
    st.markdown("## Upload Your Document", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Display the uploaded file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB"
        }
        st.write(file_details)
        
        # Process button
        if st.button("Process Document"):
            with st.spinner("Processing your document..."):
                # Step 1: Extract text
                with st.status("Step 1: Extracting text from PDF...", expanded=True) as status:
                    time.sleep(1)  # Simulate processing time
                    text = extract_text_from_pdf(uploaded_file)
                    status.update(label="Text extraction complete!", state="complete")
                
                # Step 2: Analyze structure
                with st.status("Step 2: Analyzing document structure...", expanded=True) as status:
                    time.sleep(1.5)  # Simulate processing time
                    status.update(label="Structure analysis complete!", state="complete")
                
                # Step 3: Identify key concepts
                with st.status("Step 3: Identifying key concepts...", expanded=True) as status:
                    time.sleep(1)  # Simulate processing time
                    status.update(label="Key concepts identified!", state="complete")
                
                # Step 4: Generate learning materials
                with st.status("Step 4: Generating learning module...", expanded=True) as status:
                    time.sleep(2)  # Simulate processing time
                    # Create a learning module from the text
                    module_title = uploaded_file.name.replace(".pdf", "").replace("_", " ").title()
                    learning_module = create_learning_module(text, module_title)
                    status.update(label="Learning module created!", state="complete")
                
                st.success("Document processed successfully!")
                
                # Display the learning module
                st.markdown("---")
                display_learning_module(learning_module)
    
    else:
        # Show example of the transformation process
        st.markdown("## How Unstructured Data Becomes Structured Learning", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Before: Unstructured PDF")
            st.image("https://img.icons8.com/color/96/000000/pdf.png", width=50)
            st.markdown("""
            - Raw text without clear organization
            - No interactive elements
            - Difficult to navigate and learn from
            - No way to test understanding
            - Static content that doesn't adapt
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### After: Structured Learning Module")
            st.image("https://img.icons8.com/color/96/000000/learning.png", width=50)
            st.markdown("""
            - Organized into logical sections
            - Key concepts highlighted
            - Interactive elements for engagement
            - Built-in knowledge checks
            - Estimated reading time and progress tracking
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Process visualization
        st.markdown("## The Transformation Process", unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">1</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-content">', unsafe_allow_html=True)
        st.markdown("**Text Extraction**")
        st.markdown("The system extracts raw text from the PDF document, preserving as much of the original formatting as possible.")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">2</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-content">', unsafe_allow_html=True)
        st.markdown("**Structure Analysis**")
        st.markdown("The text is analyzed to identify sections, headings, paragraphs, and other structural elements.")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">3</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-content">', unsafe_allow_html=True)
        st.markdown("**Content Classification**")
        st.markdown("Each section is classified based on its content (e.g., introduction, theory, examples, conclusion).")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">4</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-content">', unsafe_allow_html=True)
        st.markdown("**Key Concept Extraction**")
        st.markdown("Important terms, definitions, and concepts are identified and highlighted.")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="step-card">', unsafe_allow_html=True)
        st.markdown('<div class="step-number">5</div>', unsafe_allow_html=True)
        st.markdown('<div class="step-content">', unsafe_allow_html=True)
        st.markdown("**Learning Material Generation**")
        st.markdown("Interactive elements, quizzes, and summaries are generated to enhance the learning experience.")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Sample output
        st.markdown("## Sample Output", unsafe_allow_html=True)
        st.markdown("Here's an example of what a processed document looks like:")
        
        with st.expander("View Sample Learning Module"):
            st.image("https://img.icons8.com/color/96/000000/learning.png", width=50)
            st.markdown("### Introduction to Machine Learning")
            st.markdown("**Reading Time:** 15 minutes | **Sections:** 5 | **Quiz Questions:** 4")
            
            st.markdown("#### Key Concepts:")
            concepts_html = " • ".join([f"<span style='background-color: #e3f2fd; padding: 3px 8px; border-radius: 10px; margin-right: 5px;'>{concept}</span>" for concept in ["Supervised Learning", "Classification", "Regression", "Overfitting", "Validation"]])
            st.markdown(concepts_html, unsafe_allow_html=True)
            
            st.markdown("#### Section Preview:")
            st.markdown("""
            **1. What is Machine Learning?**
            
            Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. The process of learning begins with observations or data, such as examples, direct experience, or instruction...
            
            **2. Types of Machine Learning Algorithms**
            
            Machine learning algorithms are typically classified into three broad categories: supervised learning, unsupervised learning, and reinforcement learning...
            """)

if __name__ == "__main__":
    main() 