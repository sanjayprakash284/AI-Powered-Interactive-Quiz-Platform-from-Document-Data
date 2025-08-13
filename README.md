# AI-Powered-Interactive-Quiz-Platform-from-Document-Data
# PDF Quiz Generator Web App

A web application that extracts text from PDF documents and automatically generates quizzes using AI.  
Built with **Python**, **Streamlit**, and **OpenAI API**, this tool helps in creating educational quizzes from study materials, reports, or any PDF content.

---

## 🚀 Features

- **PDF Upload** – Upload any PDF file to process.
- **Text Extraction** – Extracts text content from the uploaded PDF.
- **AI-powered Quiz Generation** – Uses OpenAI API to create relevant multiple-choice or short-answer questions.
- **Interactive Interface** – Built with Streamlit for a clean and responsive UI.
- **Export Option** – Download generated quiz as a text file.

---

## 🛠 Tech Stack

- **Frontend & Backend**: [Streamlit](https://streamlit.io/)
- **AI Model**: [OpenAI API](https://platform.openai.com/)
- **PDF Processing**: [PyPDF2](https://pypi.org/project/PyPDF2/) or [pdfplumber](https://pypi.org/project/pdfplumber/)
- **Python Version**: 3.10+

---

## 📂 Project Structure

```

pdf-quiz-app/
│
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── sample.pdf            # Example PDF file (optional)

````

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/pdf-quiz-app.git
   cd pdf-quiz-app
````

2. **Create a virtual environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate      # On macOS/Linux
   venv\Scripts\activate         # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API Key**

   * Create a `.env` file in the project directory:

     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   * Or set it directly in your environment variables.

5. **Run the application**

   ```bash
   streamlit run app.py
   ```

---

## 📜 Example Usage

1. Upload a PDF file.
2. The app extracts text.
3. The AI generates quiz questions based on the extracted content.
4. View and download your quiz.

---

## 📦 Dependencies

Example `requirements.txt`:

```
streamlit
openai
PyPDF2
python-dotenv
```

---

## 📌 Future Improvements

* Support for images & diagrams in quizzes.
* Multi-language support.
* Custom quiz difficulty levels.
* Integration with Google Docs & Word files.

---



**Author:** Sanjay Prakash
📧 Contact: [chsanju647@gmail.com](chsanju647@gmail.com)

