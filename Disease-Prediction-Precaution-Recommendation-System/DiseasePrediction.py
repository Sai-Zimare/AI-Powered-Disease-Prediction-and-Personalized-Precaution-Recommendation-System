import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import zipfile
import os
import sqlite3
from googletrans import Translator, LANGUAGES

# --- FILE PATHS ---
zip_path = 'd:/Image Proccessing/archive.zip'
csv_filename = 'DiseaseAndSymptoms.csv'
precaution_csv_filename = 'Disease precaution.csv'
model_path = 'd:/Image Proccessing/disease_prediction_model.pkl'
mlb_path = 'd:/Image Proccessing/mlb.pkl'

# --- UNZIP FILES ---
if not os.path.exists(csv_filename) or not os.path.exists(precaution_csv_filename):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

# --- LOAD DATASETS ---
df = pd.read_csv(csv_filename)
df_precautions = pd.read_csv(precaution_csv_filename)

symptom_cols = [col for col in df.columns if col.startswith("Symptom")]
precaution_cols = [col for col in df_precautions.columns if col.startswith("Precaution")]

df['All_Symptoms'] = df[symptom_cols].values.tolist()
df['All_Symptoms'] = df['All_Symptoms'].apply(lambda x: [str(s).strip().lower() for s in x if pd.notna(s)])
df_precautions['All_Precautions'] = df_precautions[precaution_cols].values.tolist()
df_precautions['All_Precautions'] = df_precautions['All_Precautions'].apply(lambda x: [str(s).strip().lower() for s in x if pd.notna(s)])

# --- MODEL SETUP ---
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df['All_Symptoms'])
y = df['Disease']

if not os.path.exists(model_path) or not os.path.exists(mlb_path):
    model = MultinomialNB()
    model.fit(X, y)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(mlb_path, 'wb') as f:
        pickle.dump(mlb, f)
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(mlb_path, 'rb') as f:
        mlb = pickle.load(f)

# --- FUNCTIONS ---
def get_dynamic_severity_order():
    disease_counts = df['Disease'].value_counts()
    severity_order = {disease: rank + 1 for rank, disease in enumerate(disease_counts.index)}
    return severity_order

def predict_disease(symptoms):
    symptoms = [s.strip().lower() for s in symptoms]
    known_symptoms = set(mlb.classes_)
    filtered = [s for s in symptoms if s in known_symptoms]
    if not filtered:
        return "❌ No valid symptoms entered."
    encoded_input = mlb.transform([filtered])
    prediction = model.predict(encoded_input)
    return prediction[0]

def get_possible_diseases(symptom):
    matching_diseases = []
    for _, row in df.iterrows():
        if symptom in row['All_Symptoms']:
            matching_diseases.append(row['Disease'])
    severity_order = get_dynamic_severity_order()
    matching_diseases.sort(key=lambda x: severity_order.get(x.lower(), 999))
    return matching_diseases

def get_precautions(disease):
    disease = disease.strip().lower()
    for _, row in df_precautions.iterrows():
        if row['Disease'].strip().lower() == disease:
            return row['All_Precautions']
    return []

def create_db():
    conn = sqlite3.connect('user_queries.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symptoms TEXT,
            disease TEXT
        )
    """)
    conn.commit()
    conn.close()

def store_user_query(symptoms, disease):
    conn = sqlite3.connect('user_queries.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (symptoms, disease) VALUES (?, ?)", (', '.join(symptoms), disease))
    conn.commit()
    conn.close()

translator = Translator()

def detect_language(text):
    try:
        return translator.detect(text).lang
    except Exception as e:
        print(f"Language detection error: {e}")
        return 'en'

def translate_input(symptoms, target_lang='en'):
    try:
        return [translator.translate(symptom, src='auto', dest=target_lang).text for symptom in symptoms]
    except Exception as e:
        print(f"Translation error: {e}")
        return symptoms

def translate_output(text, target_lang):
    try:
        return translator.translate(text, src='auto', dest=target_lang).text
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# --- GUI ---
def run_gui():
    theme = {'bg': '#ffffff', 'fg': '#000000'}
    last_results = ['']

    def toggle_theme():
        theme['bg'], theme['fg'] = ('#222222', '#ffffff') if theme['bg'] == '#ffffff' else ('#ffffff', '#000000')
        apply_theme()
        update_result(last_results[0])

    def apply_theme():
        for widget in [title, symptom_title, input_label]:
            widget.configure(bg=theme['bg'], fg=theme['fg'])
        for widget in [root, search_frame, input_frame, container, inner_frame, result_frame]:
            widget.configure(bg=theme['bg'])
        symptom_entry.configure(bg='white' if theme['bg'] == '#ffffff' else '#444444', fg=theme['fg'])
        result_text.configure(bg=theme['bg'], fg=theme['fg'])

    all_symptoms = sorted(set([sym for sublist in df['All_Symptoms'] for sym in sublist]))

    def on_predict():
        entered_text = symptom_entry.get()
        symptoms = [s.strip() for s in entered_text.split(',') if s.strip()]
        if not symptoms:
            messagebox.showwarning("Input Error", "Please enter at least one symptom.")
            return

        # Detect language of the symptoms entered by the user
        user_lang = detect_language(entered_text)

        # Translate the symptoms to English for prediction
        translated_symptoms = translate_input(symptoms, target_lang='en')

        result_output = ""
        if len(symptoms) == 1:
            possible_diseases = get_possible_diseases(translated_symptoms[0].strip().lower())
            if not possible_diseases:
                result_output += "❌ No diseases found for this symptom.\n"
            else:
                for disease in set(possible_diseases):
                    precautions = get_precautions(disease)
                    if precautions:
                        result_output += f"🔍 {disease} — 🩺 Precautions: {', '.join(precautions)}\n\n"
                    else:
                        result_output += f"🔍 {disease} — ⚠️ No precautions found.\n\n"
        else:
            result = predict_disease(translated_symptoms)
            store_user_query(symptoms, result)
            precautions = get_precautions(result)
            if precautions:
                result_output += f"🔍 Predicted Disease: {result} — 🩺 Precautions: {', '.join(precautions)}"
            else:
                result_output += f"🔍 Predicted Disease: {result} — ⚠️ No precautions found."

        # Translate the result back to the user's language
        result_output = translate_output(result_output, target_lang=user_lang)
        update_result(result_output)

    def update_result(result_text_content):
        last_results[0] = result_text_content
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, result_text_content)
        result_text.config(state=tk.DISABLED)

    def refresh_page():
        symptom_entry.delete(0, tk.END)
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.config(state=tk.DISABLED)
        update_symptom_list(all_symptoms)

    def update_search_results(*args):
        search_term = search_var.get().strip()
        if search_term:
            translated_term = translate_input([search_term], target_lang='en')[0].lower()
            filtered = [s for s in all_symptoms if translated_term in s.lower()]
            update_symptom_list(filtered)

    def update_symptom_list(symptoms):
        for widget in inner_frame.winfo_children():
            widget.destroy()
        for i, s in enumerate(symptoms):
            row, col = divmod(i, max_cols)
            label = tk.Label(inner_frame, text=s, font=("Arial", 10), padx=10, pady=5, bg=theme['bg'], fg=theme['fg'])
            label.grid(row=row, column=col, padx=25, pady=8, sticky="w")

    root = tk.Tk()
    root.title("Disease Prediction and Precaution  Recommendation System")
    root.geometry("1200x750")

    title = tk.Label(root, text="Disease Prediction and Precaution  Recommendation System", font=("Helvetica", 18, "bold"))
    title.pack(pady=10)

    theme_button = tk.Button(root, text="Toggle Theme", command=toggle_theme)
    theme_button.place(x=1070, y=10)

    symptom_title = tk.Label(root, text="Symptoms", font=("Helvetica", 16, "bold"))
    symptom_title.pack(pady=(20, 5))

    refresh_button = tk.Button(root, text="Refresh", command=refresh_page, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=20, pady=10)
    refresh_button.place(x=10, y=10)

    search_frame = tk.Frame(root)
    search_frame.pack(pady=5)
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var, width=50, font=("Arial", 12))
    search_entry.pack(side="left", padx=10)
    search_entry.bind("<KeyRelease>", update_search_results)

    container = tk.Frame(root)
    container.pack(pady=10, expand=True)
    canvas = tk.Canvas(container, width=1000, height=320)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    max_cols = 5
    update_symptom_list(all_symptoms)

    input_frame = tk.Frame(root)
    input_frame.pack(pady=30)
    input_label = tk.Label(input_frame, text="Enter symptoms (comma separated):", font=("Arial", 12))
    input_label.pack()
    symptom_entry = tk.Entry(input_frame, width=70, justify="center")
    symptom_entry.pack(pady=5)

    predict_btn = tk.Button(input_frame, text="Predict Disease", font=("Arial", 12), command=on_predict, bg="#4CAF50", fg="white", padx=20, pady=5)
    predict_btn.pack(pady=10)

    result_frame = tk.Frame(root)
    result_frame.pack(pady=10, fill="both", expand=True)

    result_scrollbar = ttk.Scrollbar(result_frame, orient="vertical")
    result_scrollbar.pack(side="right", fill="y")

    global result_text
    result_text = tk.Text(result_frame, wrap="word", font=("Arial", 12), height=12, padx=20, pady=10, state=tk.DISABLED)
    result_text.pack(fill="both", expand=True)
    result_text.config(yscrollcommand=result_scrollbar.set)
    result_scrollbar.config(command=result_text.yview)

    apply_theme()
    root.mainloop()

# --- MAIN ---
if __name__ == "__main__":
    create_db()
    run_gui()
