# Progres Akurasi Model (Lokal CV)

rekap hasil coba-coba model pake data train kemarin:

- **Logistic Regression (Raw Columns)**: 35.3% (Awal banget pas belum diapa-apain)
- **Logistic Regression (+ Aggregations)**: 40.3% (Naik pas ditambah fitur mean, std, dll)
- **XGBoost (Raw Columns)**: 44.2% (Ganti algoritma langsung lumayan ngangkat)
- **XGBoost (+ Aggregations)**: 45.2% (Skor paling optimal sementara ini)

Sekarang lagi lanjut eksperimen nambahin fitur kustom baru (rasio tugas & interaksi motivasi).