import streamlit as st
import json
import pandas as pd
import numpy as np

# venv  https://code.visualstudio.com/docs/python/environments
# env >> & d:/Skripsi/deployment/.venv/Scripts/Activate.ps1
# run >> streamlit run 1_Halaman_Utama.py
# st emoji = https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app

st.set_page_config(
  page_title="Virus Segmentation App",
  page_icon="✳"
)
# st.title('YOLOv7 Virus Segmentation')
st.markdown("<span class='css-10trblm eqr7zpz0'><h1 style='text-align: center;'>YOLOv7 Virus Segmentation</h1></span>", unsafe_allow_html=True)

# ====================================================
# CSS inject : Center image when clicked
st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
#  CSS inject : remove index on table
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            tbody tr td {text-align: center;}
            </style>
            """
# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

def card_component(num, desc):
  wch_colour_box = (160, 160, 164)
  wch_colour_font = (0,0,0)
  fontsize = 28
  iconname = ""
  sline = desc
  i = num

  htmlstr = f"""<p style='background-color: rgb({wch_colour_box[0]}, 
                                                {wch_colour_box[1]}, 
                                                {wch_colour_box[2]}, 0.9); 
                          color: rgb({wch_colour_font[0]}, 
                                    {wch_colour_font[1]}, 
                                    {wch_colour_font[2]}, 0.75); 
                          font-size: {fontsize}px; 
                          border-radius: 7px; 
                          padding-left: 12px; 
                          padding-top: 18px; 
                          padding-bottom: 18px; 
                          line-height:25px;'>
                          <i class='{iconname} fa-xs'></i> {i}
                          </style><BR><span style='font-size: 14px; 
                          margin-top: 0;'>{sline}</style></span></p>"""
  return htmlstr
# ===============================================

st.markdown("""
  <p style='text-align: center; font-size:16px'>
    Dengan banyaknya jenis virus yang telah terindentifikasi dan kemungkinan terjadinya mutasi, menyebabkan pekerjaan
    pengenalan jenis virus menjadi hal yang tidak mudah, untuk mengurangi resiko kesalahan identifikasi virus
    pemanfaatan machine learning digunakan untuk membantu hal tersebut.
  </p>
""", unsafe_allow_html=True)

st.markdown("""
  <p style='text-align: center; font-size:16px'>
    🔹🔹🔷🔹🔹
  </p>
""", unsafe_allow_html=True)

st.markdown("""
  <p style='text-align: center; font-size:16px'>
    YOLOv7 merupakan algoritma pendeteksi objek yang dipublikasi oleh Wang, dkk pada 6 Juli 2022, Yolov7 merupakan
    pengembangan dari algoritma yolo sebelumnya dengan menggunakan konsep baru sepert E-ELAN, Trainable Bag of Freebies,
    dan lain-lain. 
  </p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 4, 1])
col2.image('./images/others/yolov7-performance.png', use_column_width=True)

st.markdown("""
  <p style='text-align: center; font-size:16px'>
    YOLOv7 menghasilkan performa yang melebihi algoritma pendeteksian objek lainnya 
    pada kecepatan dan akurasi dalam jarak 5 FPS sampai 160 FPS dan mempunyai akurasi tertinggi yaitu 56.8% AP 
    diantara semua algoritma pendeteksi objek lainnya dengan 30 FPS atau lebih. 
    YOLOv7-E6 objek detektor mengalahkan transformer-based detector SWIN-L, Cascade-Mask R-CNN, YOLOR, YOLOX, dll. 
    Dataset yang digunakan untuk pelatihan ialah MS COCO dataset dengan tanpa menggunaan pretrained weight. 
  </p>
""", unsafe_allow_html=True)

st.markdown("""
  <p style='text-align: center; font-size:16px'>
    🔸🔸🔶🔸🔸
  </p>
""", unsafe_allow_html=True)

st.markdown("""
  <p style='text-align: center; font-size:16px'>
    Berdasarkan dari ekperimen yang dilakukan pada 2100 data virus yang terbagi menjadi 15 kelas dengan menggunakan arsitektur YOLOv7 menghasilkan akurasi yang cukup memuaskan,
    dalam 100 iterasi yang menghabiskan waktu ±2,5 jam model mampu mensegmentasikan virus dengan error yang cukup kecil.
  </p>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.markdown(card_component(0.01662, 'box_loses'), unsafe_allow_html=True)
col2.markdown(card_component(0.0148, 'seg_loss'), unsafe_allow_html=True)
col3.markdown(card_component(0.01604, 'obj_loss'), unsafe_allow_html=True)
col4.markdown(card_component(0.00321, 'cls_loss'), unsafe_allow_html=True)

# col1, col2, col3 = st.columns([1, 4, 1])
# col2.image('./images/others/results_a.png',  use_column_width=True)
# col1, col2, col3 = st.columns([1, 4, 1])
# col2.image('./images/others/results_b.png',  use_column_width=True)

df_eval_box = pd.DataFrame(np.array([['Box', 0.904,0.91,0.947,0.788], ['Mask', 0.902,0.909,0.943,0.759]]),
                   columns=['Tipe', 'Precision', 'Recall', 'mAP50', 'mAP50-95'])
st.table(df_eval_box)

# st.markdown("""
#   <p style='text-align: center; font-size:16px'>
#     🔹🔹🔸🔷🔸🔶🔸🔷🔸🔹🔹
#   </p>
# """, unsafe_allow_html=True)