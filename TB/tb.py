import tkinter as tk
from tkinter import ttk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Direktori tempat file dokumen berada
direktori_dokumen = r'C:\Users\ASUS\OneDrive\Documents\Dataset Cerpen Online 2016\cerpen Dongeng'

# Fungsi untuk membaca semua file teks dalam direktori
def baca_dokumen_dari_direktori(direktori):
    dokumen_list = []
    file_names = []
    
    # Iterasi melalui setiap file di dalam folder
    for filename in os.listdir(direktori):
        if filename.endswith(".txt"):  # Membaca hanya file .txt
            file_path = os.path.join(direktori, filename)
            try:
                # Membuka file dengan encoding latin-1 dan mengabaikan error karakter
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
                    konten = file.read()
                    dokumen_list.append(konten)
                    file_names.append(filename)
            except Exception as e:
                print(f"Error membaca file {filename}: {e}")
    return dokumen_list, file_names

# Membaca dokumen dari folder yang ditentukan
dokumen_list, file_names = baca_dokumen_dari_direktori(direktori_dokumen)

# Membuat jendela utama
root = tk.Tk()
root.title("Pencarian Dokumen")

# Membuat label "Pertanyaan"
pertanyaan_label = tk.Label(root, text="Pertanyaan", font=("Arial", 14))
pertanyaan_label.grid(row=0, column=0, sticky="w")

# Membuat kotak teks untuk input pertanyaan
pertanyaan_entry = tk.Entry(root, width=50)
pertanyaan_entry.grid(row=0, column=1, columnspan=2, sticky="w")

# Membuat tombol "Cari"
cari_button = tk.Button(root, text="Cari", bg="red", fg="white", font=("Arial", 12))
cari_button.grid(row=0, column=3, sticky="e")

# Membuat tabel untuk menampilkan hasil pencarian
tabel_header = ["Rank", "Dokumen", "Cosinus", "Score Similarity"]
tabel = ttk.Treeview(root, columns=tabel_header, show="headings")
tabel.heading("Rank", text="Rank")
tabel.heading("Dokumen", text="Dokumen")
tabel.heading("Cosinus", text="Cosinus")
tabel.heading("Score Similarity", text="Score Similarity")
tabel.grid(row=1, column=0, columnspan=4)

# Fungsi untuk menghitung cosine similarity dan memperbarui tabel
def cari_dokumen():
    pertanyaan = pertanyaan_entry.get()
    
    # Menggabungkan pertanyaan dengan dokumen untuk TF-IDF
    corpus = [pertanyaan] + dokumen_list
    
    # Membuat TF-IDF vektor
    vectorizer = TfidfVectorizer()

    # TF-IDF Matrix
    # Pada langkah ini, setiap dokumen dalam `corpus` diubah menjadi vektor yang mewakili frekuensi
    # kata dalam dokumen (TF) dan seberapa jarang kata tersebut muncul di semua dokumen (IDF).
    # Proses ini menghasilkan sebuah matrix TF-IDF.
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Menghitung cosine similarity antara pertanyaan dan dokumen
    # Cosine similarity digunakan untuk mengukur kesamaan antara vektor pertanyaan (dokumen input)
    # dengan vektor setiap dokumen dalam kumpulan data.
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Menghapus data lama dari tabel
    for item in tabel.get_children():
        tabel.delete(item)
    
    # Mengurutkan dokumen berdasarkan cosine similarity
    sorted_indices = np.argsort(-cos_sim)  # Urutkan dari yang tertinggi
    
    # Menampilkan dokumen dengan similarity tertinggi di urutan teratas
    for rank, idx in enumerate(sorted_indices, start=1):
        similarity_score = cos_sim[idx]
        if similarity_score >= 0.8:
            score_label = "Sangat mirip"
        elif similarity_score >= 0.5:
            score_label = "Mirip"
        else:
            score_label = "Kurang mirip"
        
        tabel.insert("", tk.END, values=(rank, file_names[idx], f"{similarity_score:.2f}", score_label))
    
    # Tampilkan file dokumen yang paling mirip di urutan pertama
    file_teratas = file_names[sorted_indices[0]]
    print(f"Dokumen paling mirip: {file_teratas}")

# Menghubungkan tombol "Cari" dengan fungsi pencarian
cari_button.config(command=cari_dokumen)

root.mainloop()
