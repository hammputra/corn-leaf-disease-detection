import streamlit as st
from ultralytics import YOLO
from ultralytics.nn import tasks
from custom_layers import CBAM, ResCBAM
import __main__

__main__.CBAM = CBAM
__main__.ResCBAM = ResCBAM
tasks.CBAM = CBAM
tasks.ResCBAM = ResCBAM

from PIL import Image
import pandas as pd
import tempfile
import time

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Deteksi Penyakit Daun Jagung",
    page_icon="🌽",
    layout="wide"
)

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_models():
    model_original = YOLO("D:/corn-disease/YOLOv8.pt")
    model_rescbam = YOLO("D:/corn-disease/YOLOv8_ResCBAM.pt")
    return model_original, model_rescbam

model_original, model_rescbam = load_models()

# ======================
# FUNCTION
# ======================
def extract_detection_data(result, model):
    detections = []
    if len(result.boxes) == 0:
        return pd.DataFrame()

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "Class": class_name,
            "Confidence": round(confidence, 4),
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2)
        })
    return pd.DataFrame(detections)

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.title("🌽 Menu Navigasi")
    # Mengubah "Klasifikasi" menjadi "Deteksi"
    menu = st.radio("Pilih Halaman", ["Deteksi", "Dokumentasi", "Tentang"], label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("**Pengaturan Model**")
    conf_threshold = st.slider("Confidence Threshold", min_value=0.10, max_value=1.00, value=0.25, step=0.05)
    
    st.markdown("---")
    st.markdown("**Info Model:**\n- YOLOv8 Original\n- YOLOv8 + ResCBAM")

# Hanya jalankan fitur utama jika menu "Deteksi" dipilih
if menu == "Deteksi":
    # ======================
    # HEADER UTAMA
    # ======================
    st.title("Deteksi Penyakit Daun Jagung")
    st.markdown("Upload gambar daun jagung untuk mendeteksi lokasi dan jenis penyakit menggunakan deep learning (Common Rust, Leaf Blight, Leaf Spot).")

    # ======================
    # AREA UPLOAD & TOMBOL INFERENSI
    # ======================
    with st.container():
        st.markdown("### Upload Gambar")
        uploaded_file = st.file_uploader("Klik untuk upload atau drag & drop", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        run_button = st.button("Jalankan Inferensi", use_container_width=True, type="primary")

    st.markdown("---")
    
    # ======================
    # AREA PREVIEW & PROSES
    # ======================
    st.markdown("### Preview Gambar & Proses")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(image, caption="Gambar yang akan diproses", use_container_width=True)
            
        with col_info:
            # Mengubah judul dan deskripsi agar sesuai dengan konsep Object Detection
            # st.markdown("#### Proses Deteksi")
            # st.markdown("1. **Preprocessing**\n   Gambar diubah ukurannya agar sesuai dengan input model YOLOv8.")
            # st.markdown("2. **Deteksi Objek**\n   Model memproses gambar untuk melokalisasi (*bounding box*) dan mendeteksi jenis penyakit pada daun.")
            st.markdown("#### Status Deteksi")
            st.info("🔄 Gambar sedang diproses dan dianalisis oleh model untuk mendeteksi penyakit daun jagung.")
            if run_button:
                with st.spinner("Sedang memproses gambar..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        image.save(tmp.name)
                        image_path = tmp.name

                    # --- ORIGINAL MODEL ---
                    start_original = time.time()
                    result_original = model_original.predict(source=image_path, conf=conf_threshold, verbose=False)[0]
                    original_time = (time.time() - start_original) * 1000

                    # --- RESCBAM MODEL ---
                    start_rescbam = time.time()
                    result_rescbam = model_rescbam.predict(source=image_path, conf=conf_threshold, verbose=False)[0]
                    rescbam_time = (time.time() - start_rescbam) * 1000
                    
                    st.session_state['results'] = {
                        'res_orig': result_original,
                        'res_cbam': result_rescbam,
                        'time_orig': original_time,
                        'time_cbam': rescbam_time
                    }
                st.success("Inferensi Selesai!")

    else:
        st.info("Belum ada gambar yang dipilih. Silakan upload gambar daun jagung pada kotak di atas.")

    # ======================
    # HASIL INFERENSI
    # ======================
    if 'results' in st.session_state and uploaded_file is not None:
        st.markdown("---")
        st.markdown("### Hasil Deteksi")
        
        res_orig = st.session_state['results']['res_orig']
        res_cbam = st.session_state['results']['res_cbam']
        time_orig = st.session_state['results']['time_orig']
        time_cbam = st.session_state['results']['time_cbam']

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown("#### YOLOv8 Original")
            st.image(res_orig.plot(), use_container_width=True)

        with col_res2:
            st.markdown("#### YOLOv8 + ResCBAM")
            st.image(res_cbam.plot(), use_container_width=True)

        # SUMMARY TABLE
        st.markdown("#### Ringkasan Hasil")
        summary_df = pd.DataFrame({
            "Model": ["YOLOv8 Original", "YOLOv8 + ResCBAM"],
            "Objek Terdeteksi": [len(res_orig.boxes), len(res_cbam.boxes)],
            "Waktu Inferensi (ms)": [round(time_orig, 2), round(time_cbam, 2)]
        })
        st.dataframe(summary_df, use_container_width=True)

        # DETAIL ORIGINAL
        st.markdown("#### Detail Deteksi YOLOv8 Original")
        df_original = extract_detection_data(res_orig, model_original)
        if len(df_original) > 0:
            st.dataframe(df_original, use_container_width=True)
            csv_original = df_original.to_csv(index=False)
            st.download_button(label="Download CSV YOLOv8 Original", data=csv_original, file_name="hasil_yolov8_original.csv", mime="text/csv")
        else:
            st.warning("Tidak ada objek terdeteksi pada model YOLOv8 Original.")

        # DETAIL RESCBAM
        st.markdown("#### Detail Deteksi YOLOv8 + ResCBAM")
        df_rescbam = extract_detection_data(res_cbam, model_rescbam)
        if len(df_rescbam) > 0:
            st.dataframe(df_rescbam, use_container_width=True)
            csv_rescbam = df_rescbam.to_csv(index=False)
            st.download_button(label="Download CSV YOLOv8 ResCBAM", data=csv_rescbam, file_name="hasil_yolov8_rescbam.csv", mime="text/csv")
        else:
            st.warning("Tidak ada objek terdeteksi pada model YOLOv8 + ResCBAM.")

elif menu == "Dokumentasi":
    st.title("Dokumentasi")
    st.markdown("Panduan lengkap penggunaan aplikasi deteksi penyakit daun jagung.")

    # Menggunakan container untuk memberikan batas ruang yang rapi
    with st.container():
        st.markdown("### Pengenalan")
        st.markdown(
            "Aplikasi Deteksi Penyakit Daun Jagung adalah sistem berbasis *deep learning* yang membandingkan performa dua arsitektur: "
            "**YOLOv8 Original** dan **YOLOv8 + ResCBAM** (menggunakan modul atensi). "
            "Aplikasi ini secara otomatis melokalisasi dan mendeteksi 3 kategori penyakit pada daun jagung:"
        )
        st.markdown(
            "- Common Rust\n"
            "- Leaf Blight\n"
            "- Leaf Spot"
        )

        st.markdown("---")
        
        st.markdown("### Quick Start")
        
        st.markdown("#### 1. Upload Gambar")
        st.markdown("Navigasi ke halaman **Deteksi** melalui menu navigasi di sebelah kiri, lalu upload gambar daun jagung dengan cara:")
        st.markdown(
            "* Klik area upload *'Klik untuk upload atau drag & drop'* untuk memilih file dari komputer Anda.\n"
            "* Atau *drag & drop* (tarik dan lepas) file gambar langsung ke area upload tersebut.\n"
            "* Format yang didukung: JPG, JPEG, PNG."
        )

        st.markdown("#### 2. Atur Parameter (Opsional)")
        st.markdown(
            "Pada menu navigasi di sebelah kiri, Anda dapat mengatur **Confidence Threshold**. "
            "Ini berfungsi untuk memfilter hasil deteksi. Hanya objek dengan tingkat keyakinan (probabilitas) di atas nilai *threshold* ini yang akan ditampilkan."
        )

        st.markdown("#### 3. Jalankan Inferensi")
        st.markdown(
            "Setelah gambar ter-upload dan muncul di area *Preview*, klik tombol biru bertuliskan **Jalankan Inferensi**. "
            "Model akan mulai memproses gambar untuk mengekstraksi fitur dan mendeteksi lokasi penyakit."
        )

        st.markdown("#### 4. Lihat dan Unduh Hasil")
        st.markdown("Setelah proses selesai, hasil prediksi akan ditampilkan di bagian bawah halaman, menunjukkan:")
        st.markdown(
            "* Visualisasi *bounding box* hasil deteksi dari YOLOv8 Original dan YOLOv8 + ResCBAM.\n"
            "* Tabel ringkasan yang membandingkan jumlah objek terdeteksi dan waktu inferensi antar model.\n"
            "* Tabel detail deteksi (Kelas, Confidence, dan Koordinat).\n"
            "* Tombol untuk mengunduh tabel detail hasil deteksi ke dalam format CSV."
        )
elif menu == "Tentang":
    st.title("Tentang")
    st.markdown("Informasi tentang Mahasiswa dan Dosen Pembimbing")

    # --- CARD 1: INFORMASI PROYEK ---
    with st.container(border=True):
        st.markdown("**Implementasi Metode YOLOv8 Dengan Integrasi Resblock Dan CBAM Untuk Deteksi Penyakit Daun Jagung**")
        

        st.info(
            "**Tahun:** 2026\n\n"
            "**Institusi:** Universitas Trunojoyo Madura\n\n"
            "**Program Studi:** Teknik Informatika\n\n"
            "**Jenis Proyek:** Skripsi"
        )

    with st.container(border=True):
        st.markdown("Mahasiswa")
        st.markdown("#### Ilham Syah Putra")
        st.markdown("NIM: 220411100137")

    with st.container(border=True):
        st.markdown("Dosen Pembimbing")
        
        # Pembimbing 1
        st.markdown("#### Prof. Dr. Rima Tri Wahyuningrum, S.T., M.T.")
        st.markdown("Pembimbing 1")
        st.markdown("NIP: 198008202003122001")
        
        st.write("")
        
        # Pembimbing 2
        st.markdown("#### Dr. Cucun Very Angkoso, S.T., M.T.")
        st.markdown("Pembimbing 2")
        st.markdown("NIP: 197802252005011001")