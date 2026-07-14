# Datathon Playground 2026

Project kolaborasi untuk kompetisi Datathon 2026.

## 📁 Struktur Folder

```
datathon-playground-2026/
├── data/                           # Shared data files (DI-TRACK DI GIT)
│   ├── train.csv                   # Data training
│   ├── test.csv                    # Data test
│   └── sample_submission.csv       # Format submission
│
├── ghani/                          # Workspace Ghani
│   ├── baseline_solution.ipynb     # Notebook utama Ghani
│   ├── experiments/                # Catatan eksperimen pribadi (GITIGNORED)
│   ├── outputs/                    # File submission/output (submission penting DI-TRACK, eksperimen GITIGNORED)
│   └── models/                     # Model yang disimpan (GITIGNORED)
│
└── pitri/                          # Workspace Pitri
    ├── baseline_solution.ipynb     # Notebook utama Pitri (BUAT SENDIRI)
    ├── experiments/                # Catatan eksperimen pribadi (GITIGNORED)
    ├── outputs/                    # File submission/output (submission penting DI-TRACK, eksperimen GITIGNORED)
    └── models/                     # Model yang disimpan (GITIGNORED)
```

## 🚀 Cara Menggunakan Project

### Untuk Ghani
1. Buka `ghani/baseline_solution.ipynb`
2. Data sudah ter-link ke `../data/` (otomatis membaca dari folder `data/`)
3. Output submission akan otomatis tersimpan di `ghani/outputs/`
4. Catatan eksperimen simpan di `ghani/experiments/`

### Untuk Pitri
1. **PENTING**: Kerjakan semua pekerjaanmu di folder `pitri/`
2. Copy atau buat notebook baru di `pitri/baseline_solution.ipynb`
3. Gunakan path data yang sama: `../data/train.csv`, `../data/test.csv`, `../data/sample_submission.csv`
4. Simpan submission di `pitri/outputs/`
5. Simpan catatan eksperimen di `pitri/experiments/`
6. Simpan model di `pitri/models/` (jika perlu)

**Contoh path yang harus digunakan di notebook Pitri:**
```python
# Load data
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
sample_sub = pd.read_csv('../data/sample_submission.csv')

# Save submission
submission.to_csv('outputs/submission.csv', index=False)
```

## 📝 Aturan Git

### File yang DI-TRACK di GitHub:
- `data/*.csv` - Data shared
- `ghani/baseline_solution.ipynb` - Notebook Ghani
- `pitri/baseline_solution.ipynb` - Notebook Pitri (setelah dibuat)
- `ghani/outputs/submission_*.csv` - Submission penting Ghani (kecuali _exp, _test, _temp, _draft)
- `pitri/outputs/submission_*.csv` - Submission penting Pitri (kecuali _exp, _test, _temp, _draft)
- `.gitignore` - Aturan ignore
- `README.md` - Dokumentasi

### File yang TIDAK DI-TRACK (GITIGNORED):
- `**/catboost_info/` - File generated CatBoost
- `**/models/*.pkl, *.joblib, *.h5, *.pt, *.pth` - Model files
- `**/outputs/*_exp.csv, *_test.csv, *_temp.csv, *_draft.csv` - Output eksperimen
- `**/outputs/*.json` - Output JSON
- `**/experiments/` - Catatan eksperimen pribadi
- `.ipynb_checkpoints/` - Jupyter checkpoints
- `__pycache__/` - Python cache

## 🔧 Setup Environment

### Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn scipy
```

### Clone Repository
```bash
git clone https://github.com/Pitriwulandari/Pitriwulandari.git
cd Pitriwulandari
```

## 📊 Dataset

Dataset tersedia di folder `data/`:
- `train.csv` - Data training dengan label target
- `test.csv` - Data test untuk prediksi
- `sample_submission.csv` - Format submission yang benar

## 🔄 Workflow Git

### Commit dan Push Changes
```bash
# Add changes
git add .

# Commit dengan pesan yang jelas
git commit -m "Deskripsi perubahan"

# Push ke GitHub
git push origin main
```

### Pull Changes dari GitHub
```bash
git pull origin main
```

## 💡 Tips untuk Kolaborasi

1. **Jangan edit file orang lain**: Ghani jangan edit folder `pitri/`, Pitri jangan edit folder `ghani/`
2. **Data shared**: Folder `data/` adalah shared resource, jangan dihapus atau di-rename
3. **Output terpisah**: Setiap orang punya folder `outputs/` sendiri untuk submission
4. **Eksperimen pribadi**: Gunakan folder `experiments/` untuk catatan pribadi, tidak akan di-track di Git
5. **Naming convention untuk output**:
   - Gunakan nama seperti `submission_final.csv`, `submission_v1.csv` untuk file yang ingin di-share
   - Gunakan suffix `_exp`, `_test`, `_temp`, `_draft` untuk eksperimen yang tidak ingin di-track
6. **Communication**: Jika ada perubahan besar di struktur folder, diskusikan dulu

## 📌 Catatan Penting untuk Pitri

**Halo Pitri!** Ini adalah workspace kamu. Silakan:
1. Buat notebook `baseline_solution.ipynb` di folder ini
2. Gunakan path `../data/` untuk mengakses data shared
3. Simpan semua output di folder `outputs/`
4. Simpan catatan eksperimen di folder `experiments/`
5. Jangan lupa commit dan push notebook kamu ke GitHub

Folder ini adalah tempat kerja pribadimu, jadi bebas bereksperimen di sini!

## 🏆 Target

- **Baseline**: ~49% accuracy
- **Target Short-term**: 55-57%
- **Target Medium-term**: 58-62%
- **Target Long-term**: 62-65%

## 📞 Kontak

Jika ada pertanyaan atau kendala, silakan diskusikan bersama tim.

---

**Happy Coding! 🚀**
