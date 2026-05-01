# AI Outfit Matcher

An AI-powered prototype for identifying and matching individual items of clothing from a full outfit image. By uploading a photo of an outfit, the AI recognizes the individual pieces making up the ensemble and finds the closest matches within your provided database.

## Features
* **Dual Model Support:** Choose between the standard CLIP model or the highly improved DINOv2 model for feature extraction and matching.
* **Smart Matching:** Uses ChromaDB as a vector database to perform rapid similarity searches on clothing items.
* **Background Removal:** Optional background removal toggles to improve accuracy on noisy images.
* **Interactive UI:** A clean, user-friendly web interface powered by Streamlit.

---

## Installation & Setup

### 1. Clone the Repository
First, download the project to your local machine:
```bash
git clone https://github.com/jackb-22/AIOutfit-Matcher.git
cd AIOutfit-Matcher
```

### 2. Set Up a Virtual Environment
It is highly recommended to use a virtual environment to keep your dependencies organized.
```bash
python -m venv venv
```
Activate the virtual environment:
* **Windows:** `venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`

### 3. Install Requirements
The `requirements.txt` file contains a list of all the Python packages and libraries needed to run this project. Install them all at once by running:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

Before running the AI, you need to provide it with a database of clothing items to match against.

1. **Image Folders:** Provide properly formatted folders containing images of your clothing items. These folders must be named using the Item Number.
2. **Database CSV:** Provide a corresponding `database.csv` file that contains the item categories and descriptions linked to the item numbers.
3. **Labels:** Open `main.py` and alter the `LABEL_MAP` dictionary as needed to fit your specific clothing categories.

---

## Model Ingestion (Database Creation)

Once your data and `database.csv` are included in the project directory, you must process the images to build the vector database (ChromaDB). 

Run **one** of the following commands based on your preferred model:

**For the CLIP model:**
```bash
python ingest.py
```

**For the improved DINOv2 model:**
```bash
python ingest_dino.py
```

*Note: Once ChromaDB has successfully been created, you can decide whether to utilize background removal by changing the boolean value in `main.py`. Performance differences are generally negligible.*

---

## Running the Application

After your database is built, you can start the interactive Streamlit application. Run the command that corresponds to the model you ingested:

**If you used DINOv2:**
```bash
streamlit run app.py
```

**If you used CLIP:**
```bash
streamlit run app-clip.py
```

A browser window will open automatically. Upload your outfit images and watch the AI work its magic!

---

## Demonstrated Results

Here are a few of the matches I was able to achieve:

![Outfit Matcher Demo - Upload and Recognition](images/screenshot_1.png)
*After uploading an outfit, the model was successfully able to select individual items and provide relevant matches.*

![Outfit Matcher Demo - Database Matching](images/screenshot_2.png)
*The AI model was successfully able to recognize both shoes as one item, avoiding unnecessary duplicate matches, and provide relevant matches despite small image presence of the item.*
