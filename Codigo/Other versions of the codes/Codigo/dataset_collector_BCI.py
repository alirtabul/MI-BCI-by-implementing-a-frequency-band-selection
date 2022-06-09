# -*- coding: utf-8 -*-
"""
In this script, an eGUI will be created to guide each participant in the motor imagery
movements that they must do in order to collect data for the dataset appropriately.
The application will be created thanks to the streamlit library. 

@author: Ali Abdul Ameer Abbas
"""

# Import libraries.
from random import randint
import streamlit as st
from PIL import Image
import pandas as pd
import time

# Set the page configurations and layout.

st.set_page_config(
     page_title="EEG Imagery",
     page_icon="🧊",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.extremelycoolapp.com/help',
         'Report a bug': "https://www.extremelycoolapp.com/bug",
         'About': "Motor Imagery"
     }
)

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#111111"
font="sans serif"

st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://png.pngtree.com/thumb_back/fw800/background/20190830/pngtree-color-network-with-dots-on-white-background-image_310198.jpg");

         }}
         </style>
         """,
         unsafe_allow_html=True

     )

# Add a title, subheader, and other texts.

st.title("Motor Imagery BCI", anchor=None)
st.subheader("Ali Abdul Ameer Abbas")

patient = st.text_input('Participant Identification', 'Unkown')
trial = st.text_input('Trial', 'Unkown')

# Separate the page into three columns and add photos.
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<h1 style='text-align: center; color: black;'>LEFT</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='text-align: center; color: black;'>PASS</h1>", unsafe_allow_html=True)
with col3:
    st.markdown("<h1 style='text-align: center; color: black;'>RIGHT</h1>", unsafe_allow_html=True)

placeholder_image_1 = col1.empty()
placeholder_image_2 = col2.empty()
placeholder_image_3 = col3.empty()

with col1:
    img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_IZQUIERDA.png")
    placeholder_image_1.image(img3)
with col2:
    img3 = Image.open("FOTOS_ALAWI/NADA.jpg")
    placeholder_image_2.image(img3)
with col3:
    img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_DERECHA.png")
    placeholder_image_3.image(img3)

# Define some parameters, useful for saving the events in a csv file.
segundos = 3                # Time of green display of an action.
fs = 200                    # Sampling frequency.
repetitions = segundos * fs # Samples due to the seconds waited.  
storage_vector = []         # Vector for saving the markers in a csv file.

# Set the Start and Stop buttons, with the help of a flag.
flag = 0
if st.button('START',key="14"):
    flag = 1

if st.button("STOP",key="2"):
    flag = 0 

# Randomly activate an activity (Left-hand imagery, Right-hand imagery, Pass)

while(flag == 1):
    for _ in range(10):
        number = randint(1, 3)
        
    with col1:
        if number == 1:
            
            img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_IZQUIERDA_VERDE.png")
            placeholder_image_1.image(img3)
            time.sleep(segundos)
            storage_vector.extend([1] * repetitions)
            img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_IZQUIERDA.png")
            placeholder_image_1.image(img3)
            time.sleep(segundos)
            storage_vector.extend([0] * repetitions)

        else:
            img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_IZQUIERDA.png")
            placeholder_image_1.image(img3)


    with col2:
        if number == 2:
            img3 = Image.open("FOTOS_ALAWI/NADA_VERDE.jpeg")
            placeholder_image_2.image(img3)
            time.sleep(segundos)
            storage_vector.extend([2] * repetitions)
            img3 = Image.open("FOTOS_ALAWI/NADA.jpg")
            placeholder_image_2.image(img3)
            time.sleep(segundos)
            storage_vector.extend([0] * repetitions)

        else:
            img3 = Image.open("FOTOS_ALAWI/NADA.jpg")
            placeholder_image_2.image(img3)


    with col3:
        if number == 3:
            img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_DERECHA_VERDE.png")
            placeholder_image_3.image(img3)
            time.sleep(segundos)
            storage_vector.extend([3] * repetitions)
            img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_DERECHA.png")
            placeholder_image_3.image(img3)
            time.sleep(segundos)
            storage_vector.extend([0] * repetitions)
        else:
            img3 = Image.open("FOTOS_ALAWI/MANO_VACIA_DERECHA.png")
            placeholder_image_3.image(img3)
    
    # Save the markers in a csv file.
    df = pd.DataFrame(storage_vector) 
    print(df)
    # saving the dataframe 
    df.to_csv(f'{patient}_{trial}_Record.csv')



