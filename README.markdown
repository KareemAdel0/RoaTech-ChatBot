# RoaTech: AI Career-Guidance Chatbot 🤖

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence%20Transformers-NLP-blueviolet?style=flat)

RoaTech helps beginners find the tech track that fits them. The user describes their interests in free text, the bot predicts the closest track (Data Science, Web Development, and more), then builds a level-based learning roadmap with YouTube resources for each step.

---

## 📌 How It Works

1. **Understand the user:** the input is encoded into a semantic vector with the `all-MiniLM-L6-v2` **Sentence Transformer**.
2. **Predict the track:** a **Logistic Regression** classifier (tuned with 5-fold GridSearchCV) trained on ~500 labeled questions maps the embedding to a tech track.
3. **Build the roadmap:** the user picks a level (beginner / intermediate / advanced) and gets a structured roadmap.
4. **Find resources:** each roadmap step is linked to a video through the **YouTube Data API**.
5. **Handle messy input:** fuzzy matching (`difflib`) tolerates typos in track and level names.
6. **Collect feedback:** users rate the experience, and responses are stored to improve the bot.

## 📂 Structure

```
├── streamlit_app.py              # Chat interface
├── train_transformer_model.py    # Embeddings + Logistic Regression training
├── roadmaps.py                   # Roadmaps per track and level
├── datasets/
│   ├── track_training_cleaned_data.csv
│   └── feedback.csv
└── requirements.txt
```

## ▶️ Run Locally

```bash
git clone https://github.com/KareemAdel0/RoaTech-ChatBot.git
cd RoaTech-ChatBot
pip install -r requirements.txt
```

Add your YouTube API key to a `.env` file:

```
YOUTUBE_API_KEY=your-api-key
```

Train the model (creates `model_pkl_files/`), then start the app:

```bash
python train_transformer_model.py
streamlit run streamlit_app.py
```

---

Built by [Kareem Adel](https://www.linkedin.com/in/kareem-adel-mostafa/).
