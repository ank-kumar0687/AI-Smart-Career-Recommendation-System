# 🎯 AI Smart Career Recommendation System

> **An AI-powered Machine Learning system that recommends the most suitable career path for students based on their academic performance, technical skills, soft skills, certifications, projects, internships, and career preferences.**

---

## 🚀 Project Overview

Choosing the right career can be challenging for students because career decisions depend on multiple factors such as academic performance, technical abilities, communication skills, certifications, internships, projects, leadership qualities, and personal preferences.

The **AI Smart Career Recommendation System** uses **Machine Learning** to analyze a student's profile and recommend the career path that best matches their overall skills and academic background.

The system provides:

- 🎯 Personalized career recommendation
- 📊 Prediction confidence
- 📈 Top career probability analysis
- 🔍 Feature importance
- 📋 Student profile summary
- 📊 Interactive analytics dashboard
- 🤖 Machine Learning model insights

---

## ✨ Key Features

### 🎯 Career Prediction

Students enter their academic, technical, professional, and personal information, and the ML model predicts the most suitable career.

### 📊 Probability Analysis

The system displays the probability distribution across available career categories, allowing students to understand alternative career paths.

### 🔍 Feature Importance

The application provides insights into the factors that contribute to the model's prediction.

### 📈 Interactive Dashboard

A professional Streamlit dashboard provides an overview of:

- Student dataset statistics
- Career categories
- Model performance
- Career distribution
- Prediction results
- Career insights

### 💾 Download Assessment

Students can download their assessment/prediction information for future reference.

### 🧠 Machine Learning Based

The system uses a trained **Random Forest Classifier** with preprocessing and encoded features.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      Student         │
                    │       Input          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Data Validation    │
                    │    & Preprocessing   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ & Encoding Pipeline  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Random Forest ML    │
                    │       Model          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Career Prediction    │
                    │ + Probability Score  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    │ Results & Insights   │
                    └──────────────────────┘
```

---

# 🧠 Machine Learning Model

The project uses a **Random Forest Classifier** for multi-class career prediction.

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
```

### Dataset

| Property | Value |
|---|---:|
| Total Records | 100,000 |
| Training Records | 80,000 |
| Testing Records | 20,000 |
| Career Categories | 10 |
| Final Features | 522 |
| Numerical Features | 13 |
| Categorical Features | 9 |
| TF-IDF Features | 500 |
| Model | Random Forest |
| Test Accuracy | **99.77%** |

> **Note:** The reported 99.77% accuracy is the result obtained on the held-out test set used during model evaluation.

---

# 🎓 Career Categories

The system predicts one of the following career paths:

1. 🤖 AI/ML Engineer
2. 📊 Business Analyst
3. ☁️ Cloud Engineer
4. 🔐 Cyber Security Analyst
5. 📈 Data Analyst
6. 🧠 Data Scientist
7. ⚙️ DevOps Engineer
8. 💻 Software Engineer
9. 🎨 UI/UX Designer
10. 🌐 Web Developer

---

# 📋 Input Features

The recommendation system considers multiple dimensions of a student's profile.

### 👤 Personal Information

- Age
- Gender

### 🎓 Academic Information

- UG Course
- UG Specialization
- CGPA
- Master's Field

### 📜 Certifications & Experience

- Certification
- Certification Count
- Internship
- Internship Experience
- Projects Completed

### 💻 Technical & Core Skills

- Programming Skill
- Problem Solving
- Communication Skill
- Leadership

### 📊 Detailed Skill Scores

- Communication Score
- Problem Solving Score
- Leadership Score
- Creativity Score
- Teamwork Score

### 💼 Career Preferences

- Work Preference
- Preferred Work Environment

---

# 🔄 Machine Learning Pipeline

```text
Raw Student Data
       │
       ▼
Data Cleaning
       │
       ▼
Missing Value Handling
       │
       ▼
Duplicate Removal
       │
       ▼
Feature Selection
       │
       ▼
Categorical Encoding
       │
       ▼
Numerical Scaling
       │
       ▼
TF-IDF Feature Extraction
       │
       ▼
Feature Combination
       │
       ▼
Train/Test Split
       │
       ▼
Random Forest Classifier
       │
       ▼
Model Evaluation
       │
       ▼
Career Recommendation
```

---

# 🛠️ Technologies Used

### Programming Language

- 🐍 Python

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- TF-IDF
- Feature Scaling
- Label Encoding

### Data Processing

- Pandas
- NumPy

### Model Management

- Joblib

### Visualization

- Plotly
- Streamlit Charts

### Web Application

- Streamlit
- HTML
- CSS

### Development Environment

- Jupyter Notebook
- Visual Studio Code

---

# 📁 Project Structure

```text
AI-Smart-Career-Recommendation-System/
│
├── app.py
│
├── Ai_Carrer_project.ipynb
│
├── README.md
│
├── requirements.txt
│
├── .gitignore
│
├── model/
│   ├── career_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── target_encoder.pkl
│   └── selected_features.pkl
│
└── assets/
    └── logo.png
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/ank-kumar0687/AI-Smart-Career-Recommendation-System.git
```

```bash
cd AI-Smart-Career-Recommendation-System
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

---

# ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually it will be available at:

```text
http://localhost:8501
```

---

# 📊 Dashboard

The application includes a professional dashboard containing different sections.

### 🏠 Dashboard

Provides an overview of:

- Dataset size
- Career categories
- ML model
- Model accuracy
- Career insights
- System workflow

### 🎯 Career Prediction

Students provide their profile information and receive:

- Recommended career
- Prediction confidence
- Top career alternatives
- Probability distribution
- Feature importance

### 📈 Analytics

Provides data-driven insights and visualizations based on the available project dataset.

### 💡 Career Insights

Provides information about the supported career categories and their relevant skill areas.

### 🧠 Model Performance

Displays available model evaluation metrics and machine learning information.

### ℹ️ About

Contains information about the project, technology stack, and project objective.

---

# 📈 Example Prediction

A student enters:

```text
CGPA: 8.5
Programming Skill: High
Problem Solving: High
Communication Skill: High
Projects: 4
Certifications: 3
Internship Experience: Yes
Leadership: High
```

The system processes the profile and generates a recommendation such as:

```text
Recommended Career
        ↓
   AI/ML Engineer

Confidence
        ↓
     95.XX%

Alternative Careers
        ↓
Data Scientist
Software Engineer
Data Analyst
```

> The actual prediction and confidence depend on the trained model and student input.

---

# 📊 Model Evaluation

The project evaluates the trained model using a separate test dataset.

### Random Forest Result

```text
Training Samples : 80,000
Testing Samples  : 20,000

Number of Classes: 10
Features         : 522

Test Accuracy    : 99.77%
```

### Baseline

With 10 career categories, a simple balanced/random baseline would be approximately:

```text
10%
```

The trained model significantly outperformed this baseline on the project's test set.

---

# 🔐 Model Artifacts

The application uses pre-trained model artifacts stored inside the `model/` directory.

```text
career_model.pkl
        ↓
Trained Random Forest Model

scaler.pkl
        ↓
Numerical Feature Scaling

label_encoders.pkl
        ↓
Categorical Feature Encoding

target_encoder.pkl
        ↓
Career Label Decoding

selected_features.pkl
        ↓
Feature Order / Selection
```

These files allow the Streamlit application to perform predictions without retraining the model every time.

---

# 🎯 Project Objectives

The major objectives of this project are:

- Help students identify suitable career paths.
- Use Machine Learning for personalized career recommendations.
- Analyze academic and skill-based information.
- Provide explainable prediction insights.
- Display alternative career possibilities.
- Build an easy-to-use interactive web application.
- Demonstrate practical implementation of an end-to-end ML pipeline.

---

# 🌍 Real-World Applications

This system can potentially be used by:

- 🎓 Colleges & Universities
- 🏫 Career Guidance Centers
- 👨‍🎓 Students
- 💼 Placement Cells
- 🧑‍💼 Career Counselors
- 📚 EdTech Platforms
- 🏢 Recruitment & Skill Assessment Platforms

---

# 🔮 Future Enhancements

Future versions of the project can include:

- 🤖 Explainable AI using SHAP
- 📚 Personalized learning recommendations
- 🎓 Course recommendations for each career
- 💼 Job recommendations
- 📈 Career demand analysis
- 🧠 Deep Learning models
- 💬 AI Career Counselor chatbot
- 📄 Resume analysis
- 🔗 LinkedIn profile integration
- 📊 Real-time labor market data
- ☁️ Cloud deployment
- 👥 Student login and profile management

---

# ⚠️ Limitations

The recommendation should be treated as an **AI-assisted career guidance tool**, not as a definitive career decision.

Career suitability can depend on factors that may not be fully represented in the dataset, such as:

- Personal interests
- Long-term goals
- Financial circumstances
- Changing industry trends
- Individual learning preferences
- Real-world work experience

Therefore, students should combine the system's recommendation with professional career guidance and personal evaluation.

---

# 👨‍💻 Author

**Ankit Kumar**

🎓 MCA Student  
💻 Machine Learning & AI Enthusiast  
📊 Data Science | Machine Learning | Python | Streamlit

### GitHub

https://github.com/ank-kumar0687

---

# ⭐ Project Highlights

```text
╔══════════════════════════════════════╗
║     AI SMART CAREER RECOMMENDER      ║
╠══════════════════════════════════════╣
║                                      ║
║  👨‍🎓 100,000 Student Records          ║
║  🎯 10 Career Categories             ║
║  🧠 Random Forest Classifier          ║
║  📊 522 Engineered Features          ║
║  🎯 99.77% Test Accuracy             ║
║  🚀 Streamlit Interactive Dashboard  ║
║                                      ║
╚══════════════════════════════════════╝
```

---

# 📜 License

This project is developed for **educational, academic, and demonstration purposes**.

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

## 🚀 Built With Python & Machine Learning

**AI Smart Career Recommendation System — Turning Student Data into Career Insights.**