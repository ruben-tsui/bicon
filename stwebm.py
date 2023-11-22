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

# Session variables
if 'queryresults' not in st.session_state:
    st.session_state.queryresults = None

if 'page' not in st.session_state:
    st.session_state.page = 1  # starts at 1, not 0
    
if 'maxpage' not in st.session_state:
    st.session_state.maxpage = 0

if 'minpage' not in st.session_state:
    st.session_state.minpage = 1
    
if 'datasize' not in st.session_state:
    st.session_state.datasize = 0
    
if 'chunksize' not in st.session_state:
    st.session_state.chunksize = 3
    

#if 'table' not in st.session_state:
#    st.session_state.table = ''
table = 'Empty Table'


def buildTable(page):
    # Build table
    slices = st.session_state.queryresults
    datasize = st.session_state.datasize
    table = '<table width="100%">\n'
    n = st.session_state.chunksize
    for j in range(n):
        index = (page-1)*n + j
        if index >= datasize: break
        try:
            corpus, score, en, zh = slices[page-1][j].split('\t')
        except:
            continue
        table += '<tr>\n'
        table += f'<td>{corpus}</td><td colspan=2>{score}</td>\n'
        table += '</tr>\n'
        table += '<tr>\n'
        table += f'<td>{index+1}</td><td width="45%" valign="top">{en}</td><td width="50%" valign="top">{zh}</td>'
        table += '</tr>\n'
    table += '</table>'
    return table



appTitle= '國教院華英雙語索引典系統2.0β版'
sources = ('TWP', 'Patten', 'UNPC', 'FIN', 'QING', 'TWL', 'NTURegs', 'FTV', 'SAT', 'CIA',
           'NEJM', 'VOA', 'NYT', 'BBC', 'Quixote', 'Wiki')
corpus_labels = ('光華雜誌', '彭定康', '聯合國平行語料庫', '清史', '台灣法律(全國法規資料庫)', '臺大法規', '民視英語新聞', '科學人', '美國華人史', '新英格蘭醫學期刊', '美國之音', '紐約時報中文網', 'BBC', '唐吉柯德', '維基百科')

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
            .reportview-container .main .block-container{{
                padding-top: 0rem;
                padding-right: 0rem;
                padding-left: 0rem;
                padding-bottom: 0rem;}}
            .europe {
				font-family: Consolas, Arial;
                font-size: 16px;
			}
			.chinese {
				font-family: Microsoft Jhenghei;
                font-size: 20px;
			}
        </style>
'''
st.markdown(page_style, unsafe_allow_html=True)

# Sidebar
st.sidebar.subheader(appTitle)

with st.sidebar:

    #query = st.sidebar.text_input('輸入搜尋字串').strip()
    query = st.sidebar.text_area('輸入搜尋字串').strip()
    multicorpora = st.multiselect('選擇語料庫（可複選）', sources, ['TWP', 'FTV'])
    
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

    st.session_state.chunksize = st.slider("每頁筆數", 1, 10, 3)

    # Build user interface
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        first_button = st.button('First')
    with col2:
        prev_button = st.button('Prev')
    with col3:
        next_button = st.button('Next')
    with col4:
        last_button = st.button('Last')

# Navigation
if next_button:
    if st.session_state.page < st.session_state.maxpage:
        st.session_state.page += 1

if prev_button:
    if st.session_state.page > st.session_state.minpage:
        st.session_state.page -= 1
        
if first_button:
    st.session_state.page = st.session_state.minpage

if last_button:
    st.session_state.page = st.session_state.maxpage

page = st.session_state.page


n = st.session_state.chunksize   # chunk size (no. of rows per chunk)

# Logic to query database when button is pressed
divider = '-'*80 
if submit_button:

    # reset certin parameters
    st.session_state.queryresults = None
    st.session_state.page = 1  # starts at 1, not 0
    st.session_state.maxpage = 0
    st.session_state.minpage = 1
    st.session_state.datasize = 0
    #st.session_state.chunksize = 3


    #selectedCorpus = corpora.index(True)
    selectedCorpora = multicorpora
    
    if regex_search == 'Yes':

        #st.write('Regex search selected!!')
        all_results = [] # results from all corpora selected
        for c in selectedCorpora:
            selectedCorpus = sources.index(c)
            results = rs(query, c=selectedCorpus, max_matches=size, stats_only=(stats_only=="Yes"))
            #st.write(f'No. of matches found in [{c}]: {len(results)}')
            st.success(f'No. of matches found in [{c}]: {len(results)}')
            all_results.extend(results)

        datasize = len(all_results)
        slices = [all_results[i:i+n] for i in range(0, datasize, n)] 
        pagesize = len(slices) # total no. of pages available

        st.session_state.datasize = datasize
        st.session_state.maxpage = pagesize
        st.session_state.queryresults = slices
            
    else:

        all_results = [] # results from all corpora selected
        all_summaries = []
        for c in selectedCorpora:
            selectedCorpus = sources.index(c)
            results, summary = search(query, c=selectedCorpus, max_matches=size, stats_only=(stats_only=="Yes"))
            st.markdown(f'Corpus [{c}]: {len(results)} matches')
            all_results.extend(results)
            all_summaries.append(summary)

        datasize = len(all_results)
        slices = [all_results[i:i+n] for i in range(0, datasize, n)] 
        pagesize = len(slices) # total no. of pages available

        st.session_state.datasize = datasize
        st.session_state.maxpage = pagesize
        st.session_state.queryresults = slices
            

#st.write(st.session_state)

if st.session_state.queryresults != None:
    #col1a, col2a = st.columns([1, 2])
    #with col1a:
    st.markdown(f"page {st.session_state.page} of {st.session_state.maxpage}")
    #with col2a:
    #    st.slider("pages", 1, st.session_state.maxpage, st.session_state.page)
    table = buildTable(st.session_state.page)
    st.markdown(table, unsafe_allow_html=True)
    #st.markdown(all_summaries, unsafe_allow_html=True)
#table = buildTable(st.session_state.page)
#st.markdown(table, unsafe_allow_html=True)

#st.write("That's all, folks!")
#st.write(table)

