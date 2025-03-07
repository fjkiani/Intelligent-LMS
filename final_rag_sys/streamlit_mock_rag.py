import streamlit as st
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Intelligent LMS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
</style>
""", unsafe_allow_html=True)

# Mock educational content database
educational_content = {
    "courses": [
        {
            "id": 1,
            "title": "Introduction to Data Science",
            "description": "Learn the fundamentals of data science including statistics, Python programming, and data visualization.",
            "modules": ["Python Basics", "Statistical Analysis", "Data Visualization", "Machine Learning Fundamentals"],
            "resources": ["Python for Data Science Handbook", "Introduction to Statistical Learning", "Interactive Python Notebooks"],
            "difficulty": "Beginner",
            "duration": "8 weeks",
            "tags": ["data science", "python", "statistics", "visualization"]
        },
        {
            "id": 2,
            "title": "Advanced Machine Learning",
            "description": "Deep dive into machine learning algorithms, neural networks, and practical applications.",
            "modules": ["Supervised Learning", "Unsupervised Learning", "Neural Networks", "Deep Learning", "Model Deployment"],
            "resources": ["Deep Learning Book", "Hands-on Machine Learning", "Research Papers Collection"],
            "difficulty": "Advanced",
            "duration": "12 weeks",
            "tags": ["machine learning", "neural networks", "deep learning", "AI"]
        },
        {
            "id": 3,
            "title": "Natural Language Processing",
            "description": "Explore techniques for processing and analyzing text data using modern NLP approaches.",
            "modules": ["Text Preprocessing", "Word Embeddings", "Sequence Models", "Transformers", "NLP Applications"],
            "resources": ["Speech and Language Processing", "NLP with PyTorch", "Transformer Models Documentation"],
            "difficulty": "Intermediate",
            "duration": "10 weeks",
            "tags": ["NLP", "text analysis", "transformers", "language models"]
        },
        {
            "id": 4,
            "title": "Computer Vision Fundamentals",
            "description": "Learn to process and analyze visual data using computer vision techniques.",
            "modules": ["Image Processing", "Feature Extraction", "Object Detection", "Image Segmentation", "CNN Architectures"],
            "resources": ["Computer Vision: Algorithms and Applications", "Deep Learning for Computer Vision", "OpenCV Tutorials"],
            "difficulty": "Intermediate",
            "duration": "10 weeks",
            "tags": ["computer vision", "image processing", "CNN", "object detection"]
        }
    ],
    "articles": [
        {
            "id": 1,
            "title": "Understanding Transformer Models in NLP",
            "content": "Transformer models have revolutionized natural language processing by introducing self-attention mechanisms that capture contextual relationships in text data more effectively than previous approaches.",
            "author": "Dr. Emily Chen",
            "date": "2023-05-15",
            "tags": ["NLP", "transformers", "attention mechanisms", "language models"]
        },
        {
            "id": 2,
            "title": "The Future of Educational Technology",
            "content": "Intelligent Learning Management Systems are transforming education by personalizing learning experiences, providing real-time feedback, and adapting to individual student needs through AI and data analytics.",
            "author": "Prof. Michael Johnson",
            "date": "2023-07-22",
            "tags": ["education", "edtech", "personalized learning", "AI in education"]
        },
        {
            "id": 3,
            "title": "Data Visualization Best Practices",
            "content": "Effective data visualization is crucial for understanding complex datasets. This article covers key principles including choosing appropriate chart types, color theory, and designing for your audience.",
            "author": "Sarah Williams",
            "date": "2023-04-10",
            "tags": ["data visualization", "design principles", "charts", "data communication"]
        }
    ],
    "videos": [
        {
            "id": 1,
            "title": "Building Neural Networks from Scratch",
            "description": "Step-by-step tutorial on implementing neural networks using only NumPy, helping you understand the underlying mathematics and algorithms.",
            "duration": "45 minutes",
            "instructor": "Dr. Andrew Miller",
            "tags": ["neural networks", "deep learning", "python", "numpy"]
        },
        {
            "id": 2,
            "title": "Effective Data Cleaning Techniques",
            "description": "Learn practical approaches to prepare messy real-world data for analysis, including handling missing values, outliers, and inconsistent formatting.",
            "duration": "32 minutes",
            "instructor": "Lisa Rodriguez",
            "tags": ["data cleaning", "data preparation", "pandas", "data quality"]
        }
    ]
}

# Mock student data
student_data = {
    "progress": [
        {"course_id": 1, "module": "Python Basics", "completion": 100, "quiz_score": 92},
        {"course_id": 1, "module": "Statistical Analysis", "completion": 85, "quiz_score": 78},
        {"course_id": 1, "module": "Data Visualization", "completion": 60, "quiz_score": None},
        {"course_id": 1, "module": "Machine Learning Fundamentals", "completion": 10, "quiz_score": None},
        {"course_id": 3, "module": "Text Preprocessing", "completion": 100, "quiz_score": 88},
        {"course_id": 3, "module": "Word Embeddings", "completion": 75, "quiz_score": 82},
        {"course_id": 3, "module": "Sequence Models", "completion": 30, "quiz_score": None},
    ],
    "activity": [
        {"date": "2023-09-01", "time_spent": 45, "resource_type": "video", "resource_id": 1},
        {"date": "2023-09-02", "time_spent": 30, "resource_type": "article", "resource_id": 1},
        {"date": "2023-09-03", "time_spent": 60, "resource_type": "course", "resource_id": 1},
        {"date": "2023-09-04", "time_spent": 90, "resource_type": "course", "resource_id": 1},
        {"date": "2023-09-05", "time_spent": 45, "resource_type": "course", "resource_id": 3},
        {"date": "2023-09-06", "time_spent": 30, "resource_type": "article", "resource_id": 3},
        {"date": "2023-09-07", "time_spent": 60, "resource_type": "course", "resource_id": 3},
        {"date": "2023-09-08", "time_spent": 45, "resource_type": "video", "resource_id": 2},
        {"date": "2023-09-09", "time_spent": 75, "resource_type": "course", "resource_id": 3},
        {"date": "2023-09-10", "time_spent": 60, "resource_type": "course", "resource_id": 1},
    ],
    "recommendations": [
        {"type": "course", "id": 2, "reason": "Based on your interest in Machine Learning Fundamentals"},
        {"type": "article", "id": 3, "reason": "Complements your progress in Data Visualization"},
        {"type": "video", "id": 2, "reason": "Helpful for your current studies in Data Science"},
        {"type": "course", "id": 4, "reason": "Expands on your NLP knowledge with visual data processing"}
    ]
}

# Function to simulate document retrieval
def retrieve_educational_content(query, content_type=None, num_results=2):
    # Simulate processing time
    time.sleep(0.5)
    
    # Convert query to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Initialize results
    results = []
    
    # Search in courses
    if content_type is None or content_type == "courses":
        for course in educational_content["courses"]:
            # Check if query appears in title, description or tags
            if (query_lower in course["title"].lower() or 
                query_lower in course["description"].lower() or 
                any(query_lower in tag for tag in course["tags"])):
                results.append({
                    "type": "course",
                    "content": course
                })
    
    # Search in articles
    if content_type is None or content_type == "articles":
        for article in educational_content["articles"]:
            # Check if query appears in title, content or tags
            if (query_lower in article["title"].lower() or 
                query_lower in article["content"].lower() or 
                any(query_lower in tag for tag in article["tags"])):
                results.append({
                    "type": "article",
                    "content": article
                })
    
    # Search in videos
    if content_type is None or content_type == "videos":
        for video in educational_content["videos"]:
            # Check if query appears in title, description or tags
            if (query_lower in video["title"].lower() or 
                query_lower in video["description"].lower() or 
                any(query_lower in tag for tag in video["tags"])):
                results.append({
                    "type": "video",
                    "content": video
                })
    
    # If no results found, return some random content as "you might also be interested in"
    if not results:
        if content_type is None:
            # Get random content from each type
            course = random.choice(educational_content["courses"])
            article = random.choice(educational_content["articles"])
            results = [
                {"type": "course", "content": course, "suggested": True},
                {"type": "article", "content": article, "suggested": True}
            ]
        else:
            # Get random content of the specified type
            if content_type == "courses" and educational_content["courses"]:
                results = [{"type": "course", "content": random.choice(educational_content["courses"]), "suggested": True}]
            elif content_type == "articles" and educational_content["articles"]:
                results = [{"type": "article", "content": random.choice(educational_content["articles"]), "suggested": True}]
            elif content_type == "videos" and educational_content["videos"]:
                results = [{"type": "video", "content": random.choice(educational_content["videos"]), "suggested": True}]
    
    # Limit results
    return results[:num_results]

# Function to generate personalized learning insights
def generate_learning_insights(query, retrieved_content):
    # Simulate processing time
    time.sleep(1)
    
    if not retrieved_content:
        return "I don't have enough information to provide insights on that topic."
    
    # Generate insights based on query keywords and retrieved content
    if "data science" in query.lower() or "python" in query.lower() or "statistics" in query.lower():
        return f"""## Data Science Learning Path

Based on your interest in data science, I recommend focusing on these key areas:

1. **Python Programming**: Master the fundamentals of Python, particularly pandas and NumPy libraries
2. **Statistical Analysis**: Understand descriptive and inferential statistics
3. **Data Visualization**: Learn to create effective visualizations with matplotlib and seaborn
4. **Machine Learning**: Start with supervised learning algorithms

The course "{retrieved_content[0]['content']['title']}" would be an excellent starting point.

**Recommended next steps:**
- Complete the "Statistical Analysis" module in your current course
- Explore the article on data visualization best practices
- Schedule 30-45 minute daily practice sessions with Python
"""
    
    elif "machine learning" in query.lower() or "neural networks" in query.lower() or "deep learning" in query.lower():
        return f"""## Machine Learning Pathway

Your interest in machine learning suggests this learning path:

1. **Supervised Learning**: Master classification and regression algorithms
2. **Neural Networks**: Understand the architecture and mathematics behind neural networks
3. **Deep Learning Frameworks**: Get hands-on experience with PyTorch or TensorFlow
4. **Model Deployment**: Learn to deploy models to production environments

The "{retrieved_content[0]['content']['title']}" resource covers many of these topics in depth.

**Personalized recommendations:**
- Focus on completing the Neural Networks module in your current course
- Watch the "Building Neural Networks from Scratch" video tutorial
- Practice implementing at least one model from scratch
- Join our weekly AI study group on Thursdays
"""
    
    elif "nlp" in query.lower() or "natural language" in query.lower() or "text" in query.lower():
        return f"""## Natural Language Processing Focus

For your NLP journey, consider this structured approach:

1. **Text Preprocessing**: Master tokenization, stemming, and cleaning techniques
2. **Word Representations**: Learn about word embeddings like Word2Vec and GloVe
3. **Sequence Models**: Study RNNs, LSTMs, and GRUs for sequential data
4. **Transformer Models**: Understand attention mechanisms and transformer architecture

The article "{retrieved_content[0]['content']['title']}" provides excellent context on modern approaches.

**Your personalized learning plan:**
- Complete your current Word Embeddings module (currently at 75%)
- Allocate more time to the Sequence Models section where you're currently at 30%
- Review the transformer architecture article to prepare for upcoming modules
"""
    
    else:
        # Generic response for other queries
        content_title = retrieved_content[0]['content']['title']
        content_type = retrieved_content[0]['type']
        
        return f"""## Learning Resources on {query.title()}

I've found some relevant materials that match your interests:

The {content_type} "{content_title}" appears to be most relevant to your query.

**Personalized recommendations:**
- Explore the related resources in our library
- Connect with other students studying similar topics
- Schedule time in your calendar for focused learning sessions
- Consider joining relevant discussion forums to deepen your understanding

Based on your learning patterns, you tend to make the most progress when studying in 45-60 minute sessions in the morning.
"""

# Function to display course information
def display_course(course):
    st.markdown(f"### {course['title']}")
    st.markdown(f"**Difficulty:** {course['difficulty']} | **Duration:** {course['duration']}")
    st.markdown(f"{course['description']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Modules")
        for module in course['modules']:
            st.markdown(f"- {module}")
    
    with col2:
        st.markdown("#### Resources")
        for resource in course['resources']:
            st.markdown(f"- {resource}")
    
    st.markdown("---")

# Function to display article information
def display_article(article):
    st.markdown(f"### {article['title']}")
    st.markdown(f"*By {article['author']} on {article['date']}*")
    st.markdown(f"{article['content']}")
    st.markdown("---")

# Function to display video information
def display_video(video):
    st.markdown(f"### {video['title']}")
    st.markdown(f"**Duration:** {video['duration']} | **Instructor:** {video['instructor']}")
    st.markdown(f"{video['description']}")
    st.markdown("---")

# Function to display student progress
def display_student_progress():
    st.markdown("## Your Learning Progress", unsafe_allow_html=True)
    
    # Create progress dataframe
    progress_df = pd.DataFrame(student_data["progress"])
    
    # Get course names
    course_names = {course["id"]: course["title"] for course in educational_content["courses"]}
    progress_df["course_name"] = progress_df["course_id"].map(course_names)
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-value">2/4</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Courses In Progress</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        avg_completion = int(progress_df["completion"].mean())
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_completion}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Completion</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_score = int(progress_df["quiz_score"].dropna().mean())
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_score}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Quiz Score</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### Course Progress")
    
    # Group by course and calculate average completion
    course_progress = progress_df.groupby(["course_id", "course_name"])["completion"].mean().reset_index()
    
    # Create progress bars
    for _, row in course_progress.iterrows():
        st.markdown(f"**{row['course_name']}**")
        st.progress(row["completion"] / 100)
        st.markdown(f"{int(row['completion'])}% complete")
    
    # Create module progress chart
    st.markdown("### Module Completion")
    
    fig = px.bar(
        progress_df, 
        x="module", 
        y="completion", 
        color="course_name",
        labels={"module": "Module", "completion": "Completion (%)", "course_name": "Course"},
        height=400
    )
    
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display quiz scores
    st.markdown("### Quiz Performance")
    
    quiz_data = progress_df.dropna(subset=["quiz_score"]).copy()
    if not quiz_data.empty:
        fig = px.bar(
            quiz_data,
            x="module",
            y="quiz_score",
            color="course_name",
            labels={"module": "Module", "quiz_score": "Score (%)", "course_name": "Course"},
            height=400
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No quiz scores available yet.")

# Function to display learning activity
def display_learning_activity():
    st.markdown("## Your Learning Activity", unsafe_allow_html=True)
    
    # Create activity dataframe
    activity_df = pd.DataFrame(student_data["activity"])
    
    # Convert date strings to datetime
    activity_df["date"] = pd.to_datetime(activity_df["date"])
    
    # Calculate total study time
    total_time = activity_df["time_spent"].sum()
    avg_daily_time = activity_df.groupby("date")["time_spent"].sum().mean()
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_time}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total Minutes Studied</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{int(avg_daily_time)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg. Daily Minutes</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        streak = len(activity_df["date"].unique())
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{streak}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Day Streak</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create activity timeline
    st.markdown("### Daily Activity")
    
    daily_activity = activity_df.groupby("date")["time_spent"].sum().reset_index()
    
    fig = px.line(
        daily_activity,
        x="date",
        y="time_spent",
        markers=True,
        labels={"date": "Date", "time_spent": "Minutes Spent"},
        height=300
    )
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Minutes")
    st.plotly_chart(fig, use_container_width=True)
    
    # Create resource type breakdown
    st.markdown("### Resource Type Breakdown")
    
    resource_breakdown = activity_df.groupby("resource_type")["time_spent"].sum().reset_index()
    
    fig = px.pie(
        resource_breakdown,
        values="time_spent",
        names="resource_type",
        hole=0.4,
        height=300
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

# Function to display personalized recommendations
def display_recommendations():
    st.markdown("## Recommended for You", unsafe_allow_html=True)
    st.markdown("Based on your learning patterns and progress, we recommend these resources:")
    
    for rec in student_data["recommendations"]:
        if rec["type"] == "course":
            course = next((c for c in educational_content["courses"] if c["id"] == rec["id"]), None)
            if course:
                with st.expander(f"📚 Course: {course['title']}"):
                    st.markdown(f"**Why:** {rec['reason']}")
                    st.markdown(f"**Difficulty:** {course['difficulty']} | **Duration:** {course['duration']}")
                    st.markdown(f"{course['description']}")
                    
                    if st.button(f"Enroll in {course['title']}", key=f"enroll_{course['id']}"):
                        st.success(f"You've been enrolled in {course['title']}!")
        
        elif rec["type"] == "article":
            article = next((a for a in educational_content["articles"] if a["id"] == rec["id"]), None)
            if article:
                with st.expander(f"📄 Article: {article['title']}"):
                    st.markdown(f"**Why:** {rec['reason']}")
                    st.markdown(f"*By {article['author']} on {article['date']}*")
                    st.markdown(f"{article['content'][:150]}...")
                    
                    if st.button(f"Read {article['title']}", key=f"read_{article['id']}"):
                        st.info(f"Opening {article['title']}...")
        
        elif rec["type"] == "video":
            video = next((v for v in educational_content["videos"] if v["id"] == rec["id"]), None)
            if video:
                with st.expander(f"🎬 Video: {video['title']}"):
                    st.markdown(f"**Why:** {rec['reason']}")
                    st.markdown(f"**Duration:** {video['duration']} | **Instructor:** {video['instructor']}")
                    st.markdown(f"{video['description']}")
                    
                    if st.button(f"Watch {video['title']}", key=f"watch_{video['id']}"):
                        st.info(f"Playing {video['title']}...")

# Main Streamlit app
def main():
    # Sidebar navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
    st.sidebar.title("Intelligent LMS")
    
    # User profile in sidebar
    st.sidebar.markdown("### Student Profile")
    st.sidebar.markdown("**Name:** Alex Johnson")
    st.sidebar.markdown("**Program:** Data Science & AI")
    st.sidebar.markdown("**Level:** Intermediate")
    
    # Navigation options
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard", "Learning Assistant", "Course Catalog", "My Progress", "Activity Analytics"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About Intelligent LMS")
    st.sidebar.info(
        "Transforming unstructured data into actionable knowledge, "
        "enabling faster, more effective learning for students and education institutions."
    )
    
    # Main content based on selected page
    if page == "Dashboard":
        # Header
        st.markdown('<p class="main-header">Welcome to Your Intelligent Learning Dashboard</p>', unsafe_allow_html=True)
        st.markdown(
            "Transforming unstructured educational content into personalized, actionable knowledge to accelerate your learning journey."
        )
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">2</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Active Courses</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">65%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Overall Progress</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">85%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Quiz Average</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<div class="metric-value">10</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Day Streak</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Continue learning section
        st.markdown("## Continue Learning", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Introduction to Data Science")
            st.markdown("**Current Module:** Data Visualization")
            st.progress(0.6)
            st.markdown("60% complete")
            st.button("Resume Course", key="resume_ds")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Natural Language Processing")
            st.markdown("**Current Module:** Sequence Models")
            st.progress(0.3)
            st.markdown("30% complete")
            st.button("Resume Course", key="resume_nlp")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("## Recommended for You", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("📚 **Course**")
            st.markdown("Advanced Machine Learning")
            st.markdown("*Based on your interest in Machine Learning Fundamentals*")
            st.button("View Course", key="view_ml")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("📄 **Article**")
            st.markdown("Data Visualization Best Practices")
            st.markdown("*Complements your progress in Data Visualization*")
            st.button("Read Article", key="read_viz")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("🎬 **Video**")
            st.markdown("Effective Data Cleaning Techniques")
            st.markdown("*Helpful for your current studies in Data Science*")
            st.button("Watch Video", key="watch_clean")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recent activity
        st.markdown("## Recent Activity", unsafe_allow_html=True)
        
        activity_df = pd.DataFrame(student_data["activity"]).sort_values("date", ascending=False).head(5)
        activity_df["date"] = pd.to_datetime(activity_df["date"])
        
        for _, row in activity_df.iterrows():
            date_str = row["date"].strftime("%b %d, %Y")
            resource_type = row["resource_type"].capitalize()
            
            if resource_type == "Course":
                course = next((c for c in educational_content["courses"] if c["id"] == row["resource_id"]), None)
                if course:
                    st.markdown(f"**{date_str}:** Spent {row['time_spent']} minutes on **{course['title']}** course")
            
            elif resource_type == "Article":
                article = next((a for a in educational_content["articles"] if a["id"] == row["resource_id"]), None)
                if article:
                    st.markdown(f"**{date_str}:** Read article **{article['title']}** for {row['time_spent']} minutes")
            
            elif resource_type == "Video":
                video = next((v for v in educational_content["videos"] if v["id"] == row["resource_id"]), None)
                if video:
                    st.markdown(f"**{date_str}:** Watched video **{video['title']}** for {row['time_spent']} minutes")
    
    elif page == "Learning Assistant":
        st.markdown('<p class="main-header">AI Learning Assistant</p>', unsafe_allow_html=True)
        st.markdown(
            "Ask questions about your courses, get personalized learning recommendations, or explore new topics."
        )
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Content type filter
        content_type = st.selectbox(
            "Filter by content type:",
            [None, "courses", "articles", "videos"],
            format_func=lambda x: "All content" if x is None else x.capitalize()
        )
        
        # Query input
        if query := st.chat_input("Ask about courses, topics, or learning strategies..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Simulate retrieval process
                with st.status("Processing your request...", expanded=False) as status:
                    st.write("Searching educational content...")
                    retrieved_content = retrieve_educational_content(query, content_type)
                    st.write(f"Found {len(retrieved_content)} relevant resources")
                    
                    # Generate insights
                    st.write("Generating personalized learning insights...")
                    response = generate_learning_insights(query, retrieved_content)
                    
                    status.update(label="Response ready", state="complete")
                
                # Display the response
                message_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display retrieved content
                st.markdown("### Related Resources")
                
                for item in retrieved_content:
                    if "suggested" in item and item["suggested"]:
                        st.info("You might also be interested in:")
                    
                    if item["type"] == "course":
                        display_course(item["content"])
                    elif item["type"] == "article":
                        display_article(item["content"])
                    elif item["type"] == "video":
                        display_video(item["content"])
    
    elif page == "Course Catalog":
        st.markdown('<p class="main-header">Course Catalog</p>', unsafe_allow_html=True)
        st.markdown("Browse our collection of courses designed to help you master new skills.")
        
        # Search and filter
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input("Search courses", "")
        
        with col2:
            difficulty_filter = st.selectbox(
                "Difficulty",
                ["All", "Beginner", "Intermediate", "Advanced"]
            )
        
        # Display courses
        filtered_courses = educational_content["courses"]
        
        # Apply search filter
        if search_query:
            search_lower = search_query.lower()
            filtered_courses = [
                course for course in filtered_courses
                if search_lower in course["title"].lower() or
                search_lower in course["description"].lower() or
                any(search_lower in tag for tag in course["tags"])
            ]
        
        # Apply difficulty filter
        if difficulty_filter != "All":
            filtered_courses = [
                course for course in filtered_courses
                if course["difficulty"] == difficulty_filter
            ]
        
        if not filtered_courses:
            st.info("No courses match your search criteria.")
        else:
            for course in filtered_courses:
                with st.expander(f"{course['title']} ({course['difficulty']})"):
                    display_course(course)
                    
                    # Check if student is enrolled
                    is_enrolled = any(p["course_id"] == course["id"] for p in student_data["progress"])
                    
                    if is_enrolled:
                        st.success("You are currently enrolled in this course")
                        st.button(f"Continue Learning", key=f"continue_{course['id']}")
                    else:
                        st.button(f"Enroll Now", key=f"enroll_now_{course['id']}")
    
    elif page == "My Progress":
        st.markdown('<p class="main-header">My Learning Progress</p>', unsafe_allow_html=True)
        st.markdown("Track your course completion, quiz scores, and learning achievements.")
        
        # Display student progress
        display_student_progress()
        
        # Learning goals
        st.markdown("## Learning Goals", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Weekly Study Goal")
            st.markdown("**Target:** 5 hours per week")
            st.markdown("**Current:** 7.5 hours this week")
            st.progress(1.0)  # Exceeded goal
            st.markdown("✅ **Goal achieved!** You've exceeded your weekly study target.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Course Completion Goal")
            st.markdown("**Target:** Complete Data Science course by October 15")
            st.markdown("**Current progress:** 65%")
            st.progress(0.65)
            st.markdown("🔍 **On track.** Keep up the current pace to meet your goal.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Certificates and achievements
        st.markdown("## Certificates & Achievements", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("🏆 **Python Basics**")
            st.markdown("Completed on August 15, 2023")
            st.markdown("Score: 92%")
            st.button("View Certificate", key="cert_python")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("🏆 **Text Preprocessing**")
            st.markdown("Completed on September 5, 2023")
            st.markdown("Score: 88%")
            st.button("View Certificate", key="cert_text")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("🔒 **Data Visualization**")
            st.markdown("In progress (60% complete)")
            st.markdown("Unlock this certificate by completing the module")
            st.button("Continue Learning", key="continue_viz")
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif page == "Activity Analytics":
        st.markdown('<p class="main-header">Learning Activity Analytics</p>', unsafe_allow_html=True)
        st.markdown("Gain insights into your learning patterns and optimize your study habits.")
        
        # Display learning activity
        display_learning_activity()
        
        # Learning patterns
        st.markdown("## Learning Pattern Insights", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### Optimal Study Time")
            st.markdown("""
            Based on your activity patterns, you tend to be most productive during:
            - **Morning sessions** (8 AM - 11 AM)
            - **Evening sessions** (7 PM - 9 PM)
            
            Consider scheduling focused study during these times for maximum effectiveness.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="highlight">', unsafe_allow_html=True)
            st.markdown("### Learning Style")
            st.markdown("""
            Your engagement patterns suggest you learn best through:
            - **Visual content** (videos and interactive demos)
            - **Practical exercises** (coding and problem-solving)
            
            You might benefit from more hands-on projects and video tutorials.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations based on activity
        st.markdown("## Personalized Recommendations", unsafe_allow_html=True)
        
        # Display recommendations
        display_recommendations()

if __name__ == "__main__":
    main()