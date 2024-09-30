# to run the following code, please ensure the following libraries have been dowloaded:
# !pip install PyPDF2
# !pip install pandas
# !pip install nltk
# !pip install spacy
# !python -m spacy download en_core_web_sm

# allows the pdfs to be dowloaded for ingestion
from google.colab import files
uploaded = files.upload()

import spacy
from spacy import displacy
import PyPDF2
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# pdf path
pdf_path = list(uploaded.keys())

# loading spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# adding entity ruler
ruler = nlp.add_pipe("entity_ruler")

# example of anxiety and depression symptoms
anxiety_depression_symptoms = [
    'fatigue', 'insomnia', 'irritability', 'restlessness',
    'difficulty concentrating', 'headache', 'nausea',
    'panic attacks', 'heart palpitations', 'sweating',
    'trembling'
]

# example of common drugs that treat anxiety and depression
anxiety_depression_drugs = [
    'fluvoxamine', 'sertraline', 'paroxetine', 'escitalopram',
    'venlafaxine', 'duloxetine', 'bupropion', 'alprazolam',
    'clonazepam', 'diazepam'
]

# example of disease names
anxiety_depression_diseases = [
    'anxiety', 'depression', 'generalized anxiety disorder',
    'major depressive disorder'
]

# converting terms to patterns for EntityRuler
symptom_patterns = [
    {"label": "SYMPTOM", "pattern": [{"LOWER": term}]}
    for term in anxiety_depression_symptoms
]
drug_patterns = [
    {"label": "DRUG", "pattern": [{"LOWER": term}]}
    for term in anxiety_depression_drugs
]
disease_patterns = [
    {"label": "DISEASE", "pattern": [{"LOWER": term}]}
    for term in anxiety_depression_diseases
]

# adding patterns to entity rulers
ruler.add_patterns(drug_patterns + disease_patterns + symptom_patterns)

def extract_entities_from_pdf(pdf_path):
    """ Extracts word count from a given PDF file,
    filters out stopwords,
    and returns a DataFrame of the entity counts. """

    # initialize list for storing text
    pdf_text = []

    # open and read pdf
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)

        # extracting text from pages
        for page in reader.pages:
            text = page.extract_text()
            pdf_text.append(text)

    # combine all text into a string
    all_text = ' '.join(pdf_text)

    # applying spaCy model to the text
    doc = nlp(all_text)

    # extract namesd entities, their labels, and count them
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # filtering entities
    filtered_entities = [
        ent for ent in entities if ent[1] in
         ['PERSON', 'ORG', 'GPE', 'DRUG', 'DISEASE', 'SYMPTOM']
    ]
    entity_counter = Counter(filtered_entities)

    # convert entity counts into a DF
    df_entities = pd.DataFrame(entity_counter.items(),
                               columns=['Entity', 'Count'])

    # sort by count descending order
    df_entities = df_entities.sort_values(by='Count', ascending=False).reset_index(drop=True)

    return doc, df_entities

def split_entity_column(df):
    """
    Splits the 'Entity' column (which contains tuples) into two separate columns: 'Text' and 'Label'.
    """
    # Ensure the 'Entity' column exists and contains tuples
    if 'Entity' in df.columns:
        # Split the 'Entity' column into two separate columns: 'Text' and 'Label'
        df[['Text', 'Label']] = pd.DataFrame(df['Entity'].tolist(), index=df.index)

        # Now drop the original 'Entity' column
        df = df.drop(columns=['Entity'])

    return df

def visualize_entity_counts(df, top_n=15):
    """ Visualizes the top N entity counts in a bar plot using Seaborn. """

    # sort the DataFrame by 'Count' and get the top N entities
    df_top = df.nlargest(top_n, 'Count')

    # create a Seaborn bar plot for the top N entity counts
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Count', y='Text', hue='Label', data=df_top)

    # customize the plot
    plt.title(f'Top {top_n} Entity Frequency in Text', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Entity', fontsize=12)
    plt.tight_layout()
    plt.show()

# extract entities from uploaded pdf
doc, df_entities = extract_entities_from_pdf(pdf_path[0])

# split the 'Entity' column into 'Text' and 'Label'
df_entities = split_entity_column(df_entities)

# display the DataFrame of named entities and their counts
# df_entities.head(20)

# visualize entity counts
visualize_entity_counts(df_entities)

# second file to be scraped and visualized
# extract entities from uploaded pdf
doc, df_second_entities = extract_entities_from_pdf(pdf_path[1])

# split the 'Entity' column into 'Text' and 'Label'
df_second_entities = split_entity_column(df_second_entities)

# visualize entity counts
visualize_entity_counts(df_second_entities)

# third file to be scraped and visualized
# extract entities from uploaded pdf
doc, df_third_entities = extract_entities_from_pdf(pdf_path[2])

# split the 'Entity' column into 'Text' and 'Label'
df_third_entities = split_entity_column(df_third_entities)

# visualize entity counts
visualize_entity_counts(df_third_entities)
