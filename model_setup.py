import streamlit as st

class ModelSetup:


    @staticmethod
    def select_rank():
        #rankのレベルによってアドバイスの内容を変えていく
        rank = st.sidebar.radio("Select your rank",["初心者","中級者","上級者"])
        
        return rank



