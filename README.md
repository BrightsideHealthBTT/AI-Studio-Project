# AI-Studio-Project

Repository for our Break Through Tech Fall 2024 Studio Challenge Project with BrightSide Health

## Overview
This project extracts and visualizes relationships from clinical research papers using OpenAI GPT-4 and PyVis for creating interactive knowledge graphs. The tool is designed for tele-mental health providers to make informed decisions about treatments for anxiety and depression based on clinical research.

## Features
- Upload clinical research papers to a vector store for analysis.
- Extract structured relationships (e.g., symptoms, treatments, side effects) using OpenAI’s API.
- Generate interactive visualizations of these relationships using PyVis and NetworkX.
- Output results in a JSON format for further knowledge graph integration.

## Prerequisites
- Python
- OpenAI API Key (stored in an `.env` file)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_url>
cd AI-Studio-Project
```

### 2. Set Up a Virtual Environment
```bash
python -m venv myenv
source myenv/bin/activate   # On Windows: myenv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a new file named .env and paste the line below:
```bash
OPENAI_API_KEY= api_key_provided
```

### 5. Run the project
```bash
python main.py
```

### Key Script Components
- Upload Articles: The project uploads PDF research articles to OpenAI's vector store for processing.
- Extract Relationships: Using GPT-4, the script extracts relationships between symptoms, treatments, and medications, returning the results in JSON format.
- Visualize Relationships: The extracted relationships are visualized as an interactive graph using PyVis.

### Output

