# RoaTech Chatbot 🤖

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)  
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red.svg)

AI-powered chatbot to guide users through tech career paths with tailored roadmaps.

## 🚀 Overview

RoaTech Chatbot predicts tech tracks (e.g., Data Science) using a logistic regression model and provides learning roadmaps with YouTube resources.

## 📂 Structure
- `streamlit_app.py`: Main Streamlit app.
- `train_transformer_model.py`: Model training script.
- `track_training_cleaned_data.csv`: Training dataset.
- `requirements.txt`: Dependencies.
- `model_pkl_files/`: Trained models.

## 🛠️ Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/KareemAdel0/RoaTech_ChatBot.git
   cd RoaTech_ChatBot
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add YouTube API key in `.env`:
   ```
   YOUTUBE_API_KEY=your-api-key
   ```
5. Train the model (if needed):
   ```bash
   python train_transformer_model.py
   ```
6. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🌐 Deployment
Deploy on [Streamlit Community Cloud](https://share.streamlit.io/):
- Push to GitHub.
- Select `RoaTech_ChatBot` and `streamlit_app.py`.

## 💻 Usage
- Greet ("hello") to start.
- Choose to explore tracks or get a roadmap.
- Select a track and level (beginner/intermediate/advanced).

## 📧 Contact
Kareem Adel - [LinkedIn](https://linkedin.com/in/kareem-adel-65441a1b0)

---

*Generated on 2025-05-14 at 04:08 AM EEST*