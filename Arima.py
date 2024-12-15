import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from streamlit_option_menu import option_menu
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
from pmdarima import auto_arima

# Path ke file CSV
wisata_file_path = os.path.join('data', 'Data_Wisata.csv')

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        'Menu',
        ['Home', 'Data', 'Perhitungan ARIMA'],
        icons=['house', 'bar-chart-fill', 'plus-slash-minus'],
        menu_icon='cast',
        default_index=0
    )
st.sidebar.markdown(
    "<div style='text-align: center; font-weight: bold;'>2024 @ by Forecasting Arima Kelompok19</div>",
    unsafe_allow_html=True
)

# Home
if selected == 'Home':
    st.title('Peramalan Jumlah Perjalanan Wisatawan Nusantara ke Kota Surabaya dengan Metode ARIMA')
    st.write('Peramalan jumlah perjalanan wisatawan Nusantara ke Kota Surabaya menggunakan metode ARIMA (Autoregressive Integrated Moving Average) merupakan pendekatan yang efektif untuk menganalisis data deret waktu. Metode ini memanfaatkan pola historis data perjalanan wisatawan untuk memprediksi jumlah kunjungan di masa depan. ARIMA terdiri dari tiga komponen utama: AR (Autoregressive), yang menggambarkan hubungan antara data saat ini dan data sebelumnya; I (Integrated), yang membuat data stasioner melalui differencing; dan MA (Moving Average), yang mengaitkan data saat ini dengan error dari lag sebelumnya. Dengan memastikan data stasioner dan memilih parameter model yang tepat, ARIMA mampu memberikan prediksi yang akurat, membantu pemerintah atau pelaku industri pariwisata dalam merencanakan strategi yang lebih baik untuk mengelola kunjungan wisatawan.')

# Data Page
# Load Data Setelah Perubahan
def load_data():
    return pd.read_csv(wisata_file_path)

if selected == 'Data':
    st.title('Jumlah Perjalanan Wisatawan Nusantara ke Kota Surabaya')
    try:
        # Load Data
        data = load_data()

        st.write("Tampilkan Data Saat Ini:")
        st.write(data)

        # Input Tambah Data Baru
        with st.form(key='form_add_data'):
            st.subheader("Tambah Data Baru")
            new_time = st.text_input("Masukkan Waktu (format: YYYY-MM)", value="")
            new_value = st.text_input("Masukkan Jumlah Wisatawan", value="")
            submit_button = st.form_submit_button("Tambah Data")

        # Proses Tambah Data
        if submit_button:
            if new_time and new_value:
                new_time_parsed = pd.to_datetime(new_time, format='%Y-%m', errors='coerce')
                if new_time_parsed is not pd.NaT:
                    new_value = float(new_value.replace('.', ''))
                    new_data = pd.DataFrame({'Bulan': [new_time_parsed], 'Jumlah Wisatawan': [new_value]})
                    data = pd.concat([data, new_data], ignore_index=True)
                    data.to_csv(wisata_file_path, index=False)
                    st.success("Data berhasil ditambahkan!")
                    # Memuat ulang data setelah ditambahkan
                    data = load_data()
                    st.write(data)
                else:
                    st.error("Format waktu salah. Gunakan format: YYYY-MM.")
            else:
                st.error("Harap isi semua input.")

        # Fitur Hapus Data
        st.subheader("Hapus Data")
        selected_row = st.selectbox("Pilih Data yang Akan Dihapus:", data.index)

        if st.button("Hapus Data"):
            try:
                data = data.drop(index=selected_row).reset_index(drop=True)
                data.to_csv(wisata_file_path, index=False)
                st.success("Data berhasil dihapus!")
                # Memuat ulang data setelah dihapus
                data = load_data()
                st.write(data)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat menghapus data: {str(e)}")

    except FileNotFoundError:
        st.error("File tidak ditemukan!")

# Perhitungan ARIMA
if selected == 'Perhitungan ARIMA':
    st.title('Perhitungan ARIMA untuk Prediksi Jumlah Wisatawan ke Kota Surabaya')

    try:
        # Load Data
        data = pd.read_csv(wisata_file_path)

        # Validasi Kolom
        if 'Bulan' in data.columns and 'Jumlah Wisatawan' in data.columns:
            data['Bulan'] = pd.to_datetime(data['Bulan'], errors='coerce')
            data = data.set_index('Bulan').sort_index()

            # Tampilkan Data
            st.subheader("Data yang Digunakan:")
            st.line_chart(data['Jumlah Wisatawan'])

            # Uji Stasioneritas
            st.subheader("Uji Stasioneritas ADF Test")
            result = adfuller(data['Jumlah Wisatawan'].dropna())    
            st.write(f"ADF Statistic: {result[0]:.4f}")
            st.write(f"p-value: {result[1]:.4f}")

            # Jika data tidak stasioner, lakukan differencing
            if result[1] >= 0.05:
                st.warning("Data tidak stasioner. Akan dilakukan differencing.")

                # Differencing
                data['Jumlah Wisatawan Differenced'] = data['Jumlah Wisatawan'].diff().dropna()
                
                # Visualisasi ACF dan PACF
                st.subheader("Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF)")
                fig, ax = plt.subplots(1, 2, figsize=(16, 6))

                plot_acf(data['Jumlah Wisatawan Differenced'].dropna(), lags=20, ax=ax[0])
                ax[0].set_title("ACF - Jumlah Wisatawan Setelah Differencing")
                plot_pacf(data['Jumlah Wisatawan Differenced'].dropna(), lags=20, ax=ax[1], method='ywm')
                ax[1].set_title("PACF - Jumlah Wisatawan Setelah Differencing")

                # Visualisasi data yang telah di-differencing
                st.subheader("Data Setelah Differencing")
                st.line_chart(data['Jumlah Wisatawan Differenced'])

                # Uji Stasioneritas ulang
                diff_result = adfuller(data['Jumlah Wisatawan Differenced'].dropna())
                st.write(f"ADF Statistic (Setelah Differencing): {diff_result[0]:.4f}")
                st.write(f"p-value (Setelah Differencing): {diff_result[1]:.4f}")

                # Validasi hasil differencing
                if diff_result[1] < 0.05:
                    st.success("Data telah menjadi stasioner setelah differencing. Lanjut ke peramalan.")
                    data_to_model = data['Jumlah Wisatawan Differenced'].dropna()
                else:
                    st.error("Data tetap tidak stasioner setelah differencing. Tidak dapat melanjutkan peramalan.")
                    st.stop()
            else:
                st.success("Data sudah stasioner. Lanjut ke peramalan.")
                data_to_model = data['Jumlah Wisatawan']


            # Auto ARIMA
            st.subheader("Mencari Parameter ARIMA dengan Auto ARIMA")
            with st.spinner("Sedang mencari parameter terbaik..."):
                model_auto = auto_arima(data['Jumlah Wisatawan'], seasonal=False, trace=True, stepwise=True)

            # Parameter ARIMA
            p, d, q = model_auto.order
            st.write(f"Parameter ARIMA terbaik: (p={p}, d={d}, q={q})")

            # Input Langkah Peramalan
            steps = st.number_input("Masukkan Jumlah Periode untuk Peramalan", min_value=1, value=1, step=1)

            # Latih ARIMA
            if st.button("Lakukan Peramalan"):
                model = ARIMA(data['Jumlah Wisatawan'], order=(p, d, q))
                model_fit = model.fit()
                
                # Hasil Evaluasi Model
                st.subheader("Evaluasi Model")
                aic = model_fit.aic
                bic = model_fit.bic

                # Hitung RMSE
                residuals = model_fit.resid
                rmse = np.sqrt(np.mean(residuals**2))
                
                # Hitung MAPE (Menggunakan Fitted Values)
                y_true = data['Jumlah Wisatawan'].iloc[-len(model_fit.fittedvalues):]
                y_pred = model_fit.fittedvalues
                mape = mean_absolute_percentage_error(y_true, y_pred)

                st.write(f"**AIC (Akaike Information Criterion):** {aic:.4f}")
                st.write(f"**BIC (Bayesian Information Criterion):** {bic:.4f}")
                st.write(f"**RMSE (Root Mean Square Error):** {rmse:.4f}")
                st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape:.2f}%")

                # Visualisasi Residuals
                st.subheader("Visualisasi Residuals")
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Residuals vs Time
                axes[0].plot(residuals, label="Residuals", color='blue')
                axes[0].axhline(0, linestyle="--", color="red")
                axes[0].set_title("Residuals vs Waktu")
                axes[0].legend()

                # Histogram Residuals
                axes[1].hist(residuals, bins=20, color='blue', alpha=0.7)
                axes[1].set_title("Distribusi Residuals (Histogram)")
                axes[1].set_xlabel("Residuals")
                axes[1].set_ylabel("Frekuensi")

                # Q-Q Plot
                from scipy.stats import probplot
                probplot(residuals, dist="norm", plot=axes[2])
                axes[2].set_title("Q-Q Plot")

                st.pyplot(fig)

                # Hasil Forecast
                forecast = model_fit.forecast(steps=steps)
                forecast_index = pd.date_range(data.index[-1] + pd.DateOffset(months=1), periods=steps, freq='M')
                forecast_df = pd.DataFrame({'Jumlah Wisatawan': forecast})

                st.subheader("Hasil Peramalan:")
                st.table(forecast_df)

                # Plot Hasil Peramalan
                st.subheader("Grafik Peramalan:")
                plt.figure(figsize=(10, 6))
                plt.plot(data['Jumlah Wisatawan'], label="Data Aktual", color='blue')
                plt.plot(forecast_df['Jumlah Wisatawan'], label="Peramalan", color='red')
                plt.legend()
                plt.title("Peramalan Menggunakan ARIMA")
                st.pyplot(plt)

        else:
            st.error("Kolom 'Bulan' dan 'Jumlah Wisatawan' tidak ditemukan dalam data.")

    except Exception as e:
        st.error(f"Terjadi Kesalahan: {str(e)}")
