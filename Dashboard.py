import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder

# st.set_page_config(page_tittle='Klasifikasi' ,layout='wide')
# Memuat model dan scaler
clf = joblib.load('models/svm_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Inisialisasi session state
if 'predictions_history' not in st.session_state:
    st.session_state['predictions_history'] = []

if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None


# Fungsi untuk mengubah fitur kategorikal menjadi numerik
def encode_input_data(input_data):
    encoders = {}  # Menyimpan encoder untuk setiap kolom kategorikal
    encoded_data = input_data.copy()

    for column in input_data.columns:
        if input_data[
                column].dtype == 'object':  # Hanya kolom string/kategorikal
            le = LabelEncoder()
            encoded_data[column] = le.fit_transform(input_data[column])
            encoders[
                column] = le  # Simpan encoder jika diperlukan untuk decoding

    return encoded_data, encoders


# Fungsi prediksi
def predict_status(input_data):
    # Encode data kategorikal
    encoded_data, _ = encode_input_data(input_data)

    # Normalisasi dengan scaler
    input_data_scaled = scaler.transform(encoded_data)

    # Prediksi dengan model
    prediction = clf.predict(input_data_scaled)
    return prediction


# Membuat tab
tab1, tab2, tab3 = st.tabs(["Prediksi", "Visualisasi", "Data"])

with tab1:
    # Form input pengguna
    st.header("Masukkan Data")

    # Input nama tetap berada di atas
    name = st.text_input('Nama')

    # Membuat dua kolom
    col1, col2 = st.columns(2)

    # Input data pengguna, dibagi menjadi dua kolom
    with col1:
        status_hidup_ayah = st.selectbox("Status Hidup Ayah", ['Masih Hidup', 'Meninggal'])
        status_hidup_ibu = st.selectbox("Status Hidup Ibu", ['Masih Hidup', 'Meninggal'])
        status_kerja_ayah = st.selectbox("Status Kerja Ayah", ['Bekerja', 'Tidak Bekerja'])
        status_kerja_ibu = st.selectbox("Status Kerja Ibu", ['Bekerja', 'Tidak Bekerja'])
        tempat_tinggal = st.selectbox("Tempat Tinggal", ['Milik Sendiri', 'Ikut Saudara', 'Kontrak'])
        penghasilan_ortu = st.selectbox("Penghasilan Orang Tua", [
            '> 4 Juta', '2 Juta - 4 Juta', '1 Juta - 2 Juta', '500 Ribu - 1 Juta', '< 500 Ribu'
        ])
        kartu_pkh = st.selectbox("Kartu PKH", ['Tidak Punya', 'Punya'])
        kps = st.selectbox("KPS", ['Tidak Punya', 'Punya'])

    with col2:
        kks = st.selectbox("KKS", ['Tidak Punya', 'Punya'])
        dinding_rumah = st.selectbox("Dinding Rumah", ['Tembok', 'Kayu'])
        lantai_rumah = st.selectbox("Lantai Rumah", ['Keramik', 'Tanah'])
        kondisi_rumah = st.selectbox("Kondisi Rumah", ['Baik', 'Rusak Ringan', 'Rusak Berat'])
        jumlah_tanggungan = st.selectbox("Jumlah Tanggungan", ['1 - 2 Orang', '3 - 4 Orang', '> 5 Orang'])
        prestasi = st.selectbox("Prestasi", [
            'Tidak Pernah', 'Juara Tingkat Kabupaten', 'Juara Tingkat Provinsi', 'Juara Tingkat Nasional'
        ])
        keaktifan_belajar = st.selectbox("Keaktifan Belajar", [
            'Alpa > 14 Kali', 'Alpa 8 - 13 Kali', 'Alpa 3 - 7 Kali', 'Alpa < 2 Kali', 'Tidak Pernah'
        ])
        pelanggaran_sekolah = st.selectbox("Pelanggaran di Sekolah", ['Tidak Pernah', 'Pernah'])

    # Memasukkan data ke dalam DataFrame
    input_data = pd.DataFrame([[
        status_hidup_ayah, status_hidup_ibu, status_kerja_ayah, status_kerja_ibu,
        tempat_tinggal, penghasilan_ortu, kartu_pkh, kps, kks, dinding_rumah,
        lantai_rumah, kondisi_rumah, jumlah_tanggungan, prestasi, keaktifan_belajar, pelanggaran_sekolah
    ]], columns=[
        'Status Hidup Ayah', 'Status Hidup Ibu', 'Status Kerja Ayah', 'Status Kerja Ibu',
        'Tempat Tinggal', 'Penghasilan Orang Tua', 'Kartu PKH', 'KPS', 'KKS', 'Dinding Rumah',
        'Lantai Rumah', 'Kondisi Rumah', 'Jumlah Tanggungan', 'Prestasi', 'Keaktifan Belajar', 'Pelanggaran di Sekolah'
    ])

    # Tombol prediksi 
    if st.button("Prediksi"):
        # Validasi nama
        if name:
            prediction = predict_status(input_data)
            prediction_result = "tidak diterima" if prediction[0] == 1 else "diterima"
            input_data['Nama'] = name
            input_data['Prediction'] = prediction_result

            # Simpan hasil ke session state
            st.session_state['predictions_history'].append(input_data)
            st.session_state['prediction_result'] = prediction_result

            st.write(f"**Hasil Prediksi untuk {name}:** {prediction_result}")
        else:
            st.warning("Silakan masukkan nama.")

with tab2:
    # Fungsi untuk menampilkan visualisasi
    def tampilkan_visualisasi(prediction_result):
        labels = ['Diterima', 'Tidak Diterima']
        if prediction_result == "Status diterima":
            sizes = [80, 20]
            colors = ['#4CAF50', '#F44336']
        else:
            sizes = [20, 80]
            colors = ['#4CAF50', '#F44336']

        # Pie Chart
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90)
        ax1.axis('equal'
                 )  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Bar Chart
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, sizes, color=colors)
        ax2.set_title('Bar Chart: Diterima vs Tidak Diterima')

        # Histogram
        data = np.random.randn(1000)  # Contoh data acak untuk histogram
        fig3, ax3 = plt.subplots()
        ax3.hist(data, bins=30, color='#42A5F5', edgecolor='black')
        ax3.set_title('Histogram: Distribusi Data')

        # Line Chart
        x = np.arange(0, 10)
        y = np.random.randn(10).cumsum()
        fig4, ax4 = plt.subplots()
        ax4.plot(x, y, marker='o', color='#FF9800')
        ax4.set_title('Line Chart: Perubahan Prediksi')

        return fig1, fig2, fig3, fig4

    # Mengecek apakah prediksi sudah dilakukan
    if st.session_state['prediction_result']:
        prediction_result = st.session_state['prediction_result']

        # Tampilkan hasil prediksi
        st.write(f"**Hasil Prediksi:** {prediction_result}")

        # Menampilkan visualisasi berdasarkan hasil prediksi
        col1, col2 = st.columns(2)

        with col1:
            # Menampilkan dua visualisasi pertama (Pie Chart dan Bar Chart)
            fig1, fig2, _, _ = tampilkan_visualisasi(prediction_result)
            st.pyplot(fig1)
            st.pyplot(fig2)

        with col2:
            # Menampilkan dua visualisasi berikutnya (Histogram dan Line Chart)
            _, _, fig3, fig4 = tampilkan_visualisasi(prediction_result)
            st.pyplot(fig3)
            st.pyplot(fig4)
    else:
        st.write("⚠️ **Silakan lakukan prediksi terlebih dahulu.**")

with tab3:
    st.header("Data yang Diprediksi")

    if len(st.session_state['predictions_history']) > 0:
        all_predictions = pd.concat(st.session_state['predictions_history'],
                                    ignore_index=True)
        st.download_button("Download Predicted Data",
                           all_predictions.to_csv(index=False),
                           file_name="predicted_data.csv",
                           mime="text/csv")
        st.table(all_predictions)
    else:
        st.write("Belum ada prediksi yang dilakukan.")
