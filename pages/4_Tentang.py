import streamlit as st
import os
import base64
import numpy as np

# =====================================
# read img from local using markdown html
@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data()
def get_img_with_href(local_img_path, target_url, name):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <div style="text-align: center">
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}"/>
            <span style="align: center; display:block; font-size:11px">{name}<span>
        </a>
        </div>
        '''
    return html_code

def show_contact(contact_data):
    contact_data = np.array(contact_data)
    bin_str = [get_base64_of_bin_file(x) for x in contact_data[:,0]]
    target = contact_data[:,1]
    name = contact_data[:,2]
    styl = f"""
        <style>
            .contacts a{{ 
                text-decoration: none;
            }}
            .contacts a:hover{{
                color: #009dd1;
                display:block;
                background-color:#00bac750;
            }}
        </style>
        """
    st.markdown(styl, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='display:flex; flex-wrap:wrap; justify-content: center;'>
        <div style='flex: 1 1 auto;' class="contacts">
            <a href="{target[0]}">
                <img src="data:image/png;base64,{bin_str[0]}" style="display: block; margin-left: auto; margin-right: auto;"/>
                <span style="text-align: center; display:block; font-size:11px";>{name[0]}<span>
            </a>
        </div>
        <div style='flex: 1 1 auto;' class="contacts">
            <a href="{target[1]}">
                <img src="data:image/png;base64,{bin_str[1]}" style="display: block; margin-left: auto; margin-right: auto;"/>
                <span style="text-align: center; display:block; font-size:11px">{name[1]}<span>
            </a>
        </div>
        <div style='flex: 1 1 auto;' class="contacts">
            <a href="{target[2]}">
                <img src="data:image/png;base64,{bin_str[2]}" style="display: block; margin-left: auto; margin-right: auto;"/>
                <span style="text-align: center; display:block; font-size:11px">{name[2]}<span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =====================================

st.set_page_config(
  page_title="About - VSA",
  page_icon="✳"
)
st.title('Tentang Saya')



my_email = 'https://mail.google.com/mail/?view=cm&fs=1&to=ikhsan.adi19@mhs.uinjkt.ac.id&su=VirusSegmentationApps'
# for i in range(1,2): # number of rows in your table! = 2
sections = st.columns(2)
with sections[0]:
    st.write("""
        Aplikasi ini dibuat oleh Ikhsan Adi Putra sebagai tugas akhir perkuliahan di UIN Syarif Hidayatullah Jakarta 2023.
        """)
    st.write("""
            Saran dan masukkan anda sangat diperlukan yang dapat disampaikan melalui link berikut ini:
            """)
    show_contact([['./images/logo/gmail.png', my_email, 'Gmail'],
                ['./images/logo/linkedin.png', 'https://www.linkedin.com/in/ikhsan-adi-putra-63a2a1137/', 'LinkedIn'],
                ['./images/logo/github.png', 'https://github.com/rootidn', 'Github']])
    st.write("")
with sections[1]:
    bin_str = get_base64_of_bin_file('./images/others/study.png')
    st.markdown(f"""
    <div style='display:flex; flex-wrap:wrap'>
        <a href="https://www.freepik.com/free-vector/kids-studying-from-home-concept-illustration_7709382.htm">
            <img src="data:image/png;base64,{bin_str}" style="object-fit: cover; max-width: 100%; height: auto; vertical-align: middle;"/>
        </a>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.info('Aplikasi ini merupakan prototype sebagai gambaran aplikasi pendeteksi virus yang perlu dilakukan pengembangan lebih lanjut untuk meningkatkan kualitas hasil prediksi dan pengujian oleh para ahli dibidang yang terkait.')
