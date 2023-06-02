import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from sklearn import preprocessing
import scipy.stats as stats
import requests
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import csv
import os
from AutoClean import AutoClean
import webbrowser

st.set_page_config(layout='wide')

hide_st_style="""
<style>
footer{visibility:hidden;}
</style>"""

st.markdown(hide_st_style,unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


local_css("style.css")


st.markdown("""
<div class="container">
        <span class="text1">welcome to</span>
        <span class="text2">Data Pre-processing toolkit</span>
    </div>""",unsafe_allow_html=True)
st.markdown(" ")
st.markdown(" ")
st.markdown("***")
st.markdown('''
This is the part of **Data Pre-processing toolkit** created in Streamlit. 
**Credit:** App built in `Python` + `Streamlit` by Deepak and Nehansh.
''')
st.header('Data Cleaning functionality')
st.header('Upload your CSV file')
uploaded_file = st.file_uploader('upload input csv file')

def open_csv(dataset):
    with open('data.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        dict_from_csv = dict(list(csv_reader)[0])
        list_of_column_names = list(dict_from_csv.keys())
        return list_of_column_names

def display_output(csv):
    return st.write(csv)

def pre_process(csv,dups,missing,out,encoding,miss_cat,dt):
     file = pd.read_csv(csv)
     pipeline = AutoClean(file,mode='manual',duplicates=dups,missing_num=missing,outliers=out,encode_categ=encoding,missing_categ=miss_cat,extract_datetime=dt)
     temp = pipeline.output
     csv_file = temp.to_csv()
     st.write(pipeline.output)
    #  display_output(pipeline.output)
     return csv_file
image_aug_URI = "https://varmadeepak-image-aug-image-aug-pmo6u0.streamlit.app/"
text_eda_URI = "https://varmadeepak-textanalyzer-app-jhuzko.streamlit.app/"

# if st.button("Image_Augmentation ?"):
#          webbrowser.open_new_tab(image_aug_URI)
if st.button("Image_Augmentation"):
     st.markdown(f'<a href="{image_aug_URI}" target="_blank">Click here to open the webpage</a>', unsafe_allow_html=True)
if st.button("Text_EDA"):
    st.markdown(f'<a href="{text_eda_URI}" target="_blank">Click here to open the webpage</a>', unsafe_allow_html=True)
def download_csv_data(csv):
    return st.download_button(
                 label="Download data as CSV",
                 data= csv,
                 mime='text/csv')

if uploaded_file is not None:
    split_tup = os.path.splitext(uploaded_file.name)
    @st.cache
    def load_csv(up_file):
        csv = pd.read_csv(up_file,encoding="latin1")
        return csv
    df = load_csv(uploaded_file)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    def read_csv_file(csv):
        return pd.read_csv(csv)
    
   
    def load_lottieurl(url):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_coding = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_49rdyysj.json")

    with st.sidebar.header('**visualize**'):
        st_lottie(lottie_coding, height=300, key="coding")

        st.sidebar.header("Basic Visualization")
        if st.sidebar.checkbox("Columns Names"):
            st.sidebar.write(df.columns)

        # Show Shape of Dataset
        if st.sidebar.checkbox("Shape of Dataset"):
            st.sidebar.write(df.shape)
        
        # Show Columns By Selection
        if st.sidebar.checkbox("Select Columns To Show"):
            all_columns = df.columns.tolist()
            selected_columns = st.sidebar.multiselect('Select',all_columns)
            new_df = df[selected_columns]
            st.sidebar.dataframe(new_df)

        # Value Counts
        if st.sidebar.checkbox("Value Counts"):
            st.sidebar.text("Value Counts By Target/Class")
            st.sidebar.write(df.iloc[:,-1].value_counts())

        # Summary
        if st.sidebar.checkbox("Summary"):
            st.sidebar.write(df.describe())

    st.header("Want to analyze the data?")
    if st.button('Analyze Data ? '):
        pr = ProfileReport(df,explorative=True)
        st.header('**Pandas Profiling Report**')
        st_profile_report(pr)

    co1,co2,co3=st.columns(3)

    with co1:
        st.header("Check for any missing values")
        if st.checkbox("missing values"):
            st.write(df.isnull().sum())

    with co2:
        st.header("Check for Duplicates")
        if st.checkbox("Duplicates"):
            st.write(df[df.duplicated()])

    with co3:
        st.header("check for outliers")
        if st.checkbox("outliers"):
            
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            st.write(IQR)

    if st.header("Data Cleaning Functionalities"):
        option_clean = st.selectbox(
        'Handle all  Data Cleaning functionalities',
        ('--select option--','Yes','No'))
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
            
        if option_clean == 'Yes':
            with col1:
                option_dups = st.selectbox('Duplicates?',('auto', False))
            with col2:
                option_miss = st.selectbox('Missing Values (Numeric)?',('mean', 'median', 'most_frequent','linreg','knn',False))
            with col3:
                option_str = st.selectbox('Missing Values(String)?',('most_frequent','logreg','knn',False))
            with col4:    
                option_out = st.selectbox('Outliers?',('auto', 'winx', 'delete',False))
            with col5:
                option_enc = st.selectbox('Encode Data?',('auto', '[onehot]', '[label]',False))
            with col6:
                option_dt = st.selectbox('Datetime?',('auto', 'D', 'M','Y',False))
            if st.button("Clean"):
                csv_temp = pre_process(uploaded_file,option_dups,option_miss,option_out,option_enc,option_str,option_dt)
                download_csv_data(csv_temp)


    st.header("Data Normalization")
    
    c1,c2=st.columns(2)
    with c1:
        if st.checkbox("Min-Max Normalization"):
                df=df.select_dtypes(include='number')
                all_col = df.columns.tolist()
                selected_col = st.multiselect('Select',all_col)
                new_df = df[selected_col]
                if st.button("normalize"):
                    d = preprocessing.normalize(new_df)
            #d_df = pd.DataFrame(df, columns=index)
                    #st.write(d)
                    st.dataframe(d)
    with c2:
        if st.checkbox("Z-score Normalization"):
                df=df.select_dtypes(include='number')
                all_col = df.columns.tolist()
                selected_col1 = st.multiselect('Select',all_col)
                new_df = df[selected_col1]
                if st.button("normalize"):
                    d = stats.zscore(new_df)
            #d_df = pd.DataFrame(df, columns=index)
                    #st.write(d)
                    st.dataframe(d)
    
else:
     st.info('Awaiting for CSV file to be uploaded.')
     

