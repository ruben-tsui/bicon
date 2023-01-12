# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 19:51:02 2022

@author: ruben
"""

import streamlit as st
#from streamlit.components.v1 import html
#import streamlit.components.v1 as components
import pandas as pd
#import numpy as np
#from datetime import datetime

from biconWebStm import s as search, regex_search as rs

appTitle= '國教院華英雙語索引典系統2.0α版'
sources = ('TWP', 'Patten', 'UNPC', 'TWL', 'NTURegs', 'FTV', 'SAT', 'CIA',
           'NEJM', 'VOA', 'NYT', 'BBC', 'Quixote', 'Wiki')
corpus_labels = ('光華雜誌', '彭定康', '聯合國平行語料庫', '台灣法律(全國法規資料庫)', '臺大法規', '民視英語新聞', '科學人', '美國華人史', '新英格蘭醫學期刊', '美國之音', '紐約時報中文網', 'BBC', '唐吉柯德', '維基百科')

files = [c + '.xz' for c in sources]


st.set_page_config(
    page_title=appTitle,
    #page_icon=icon,
    layout='wide',
    initial_sidebar_state='auto',
    menu_items={
        'Get Help': 'https://streamlit.io/',
        'Report a bug': 'https://github.com',
        'About': f'**{appTitle}**\nCopyright (c) Ruben G. Tsui'
        }
)

page_style = '''
        <style>
            .css-o18uir.e16nr0p33 {
              margin-top: -125px;
            }
            .reportview-container .css-1lcbmhc .css-1outpf7 {{
                padding-top: -125px;
            }}
	       .europe {
				font-family: Consolas, Arial;
                font-size: 20px;
			}
			.chinese {
				font-family: Microsoft Jhenghei;
                font-size: 20px;
			}
        </style>
'''
st.markdown(page_style, unsafe_allow_html=True)

# Sidebar
#st.sidebar.title(appTitle)
#st.sidebar.subheader('語料庫')
st.sidebar.subheader(appTitle)

    #col0 = st.sidebar.columns(1)
    #with col0:
    #    query = st.text_input('輸入搜尋字串')
    #    submit_button = st.form_submit_button('搜尋')

with st.sidebar:

    #query = st.sidebar.text_input('輸入搜尋字串').strip()
    query = st.sidebar.text_area('輸入搜尋字串').strip()
    multicorpora = st.multiselect('Select corpora', sources, ['TWP', 'FTV'])
    
    colc, cold = st.sidebar.columns([1, 1])
    with colc:
        submit_button = st.button('搜尋')
    with cold:
        regex_search = st.radio(
            "Regex search",
            ["No", "Yes"], horizontal=True
        )
         
    
    cola, colb = st.sidebar.columns([1, 1])
    with cola:
        size = st.selectbox('筆數上限', [10,20,50,100,200,500],)
    with colb:
        stats_only = st.radio(
            "Stats only",
            ["No", "Yes"], horizontal=True
        )

    #col1, col2 = st.sidebar.columns(2)
    #corpora = []
    #n = len(sources) # no. of corpora
    #with col1:
    #    for s, h in zip(sources[:n//2], corpus_labels[:n//2]):
    #        if s == 'TWP':
    #            corpora.append(st.checkbox(label=s, value=True, help=h))
    #        else:
    #            corpora.append(st.checkbox(label=s, value=False, help=h))
    #with col2:
    #    for s, h in zip(sources[n//2:], corpus_labels[n//2:]):
    #        corpora.append(st.checkbox(label=s, value=False, help=h))



# Logic to query database when button is pressed
divider = '-'*80 
if submit_button:

    #selectedCorpus = corpora.index(True)
    selectedCorpora = multicorpora
    
    if regex_search == 'Yes':

        #st.write('Regex search selected!!')
        for c in selectedCorpora:
            selectedCorpus = sources.index(c)
            results = rs(query, c=selectedCorpus, max_matches=size, stats_only=(stats_only=="Yes"))
            #st.write(results)
            #st.write(f'No. of matches found: {len(results)}')
            st.markdown(f'Corpus [{c}]: {len(results)} matches')
            for res in results:
                st.markdown(res, unsafe_allow_html=True)
                #st.table(res, unsafe_allow_html=False)
                st.markdown(divider)

    else:

        for c in selectedCorpora:
            selectedCorpus = sources.index(c)
            results = search(query, c=selectedCorpus, max_matches=size, stats_only=(stats_only=="Yes"))
            #st.write(results)
            st.markdown(f'Corpus [{c}]: {len(results)} matches')
            for res in results:
                st.markdown(res, unsafe_allow_html=True)
                st.markdown(divider)
            
        #with st.container():
        #    
        #    for res in results:
        #        #st.write('Funny Business')
        #        #st.markdown(res, unsafe_allow_html=True)
        #        st.table(pd.DataFrame(res))
        #        st.markdown(divider)
