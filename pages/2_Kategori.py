import streamlit as st
import json

st.set_page_config(
  page_title="Kategori - VSA",
  page_icon="✳"
)
# st.title('Kategori Virus')
st.markdown("<span class='css-10trblm eqr7zpz0'><h1 style='text-align: center;'>Kategori Virus</h1></span>", unsafe_allow_html=True)

st.markdown("""
  <p style='text-align: center; font-size:18px'>
  Pada aplikasi ini terdapat 15 jenis virus yang dapat diklasifikasikan dengan menggunakan arsitektur model YOLOv7.
  </p>
""", unsafe_allow_html=True)
st.markdown("""
  <p style='text-align: center; font-size:16px'>
    🔸🔹🔷🔹🔸
  </p>
  </br>
""", unsafe_allow_html=True)
f = open('./images/viruses/viruses.json')
viruses = json.load(f)

# "0":{
#     "name":"Adenovirus",
#     "text":"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
#     "url":"./images/viruses/adenovirus.png"
#   }

for i in range(0,len(viruses)): # number of rows
  virus_data = viruses["{}".format(i)]
  cols = st.columns(2) # number of columns  = 2
  cols[0].image(virus_data['img-url'], caption= virus_data['name'], use_column_width=True)
  cols[1].markdown("""
  <h2 style='padding: 0px;'>{}) {}</h2>
  <p><i>{}</i></p>
  <p>{} (<a href='{}'>sumber</a>)</p>
  """.format(i+1, virus_data['name'], virus_data['famili'], virus_data['text'], virus_data['doi']), unsafe_allow_html=True)
  if i < len(viruses)-1:
    st.markdown("""<hr style='margin-top:5px'>""", unsafe_allow_html=True)
  # st.divider()