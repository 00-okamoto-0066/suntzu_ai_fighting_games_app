from langchain_openai import OpenAIEmbeddings
import streamlit as st
import fitz
import re
from langchain.schema import Document
from langchain_community.vectorstores import FAISS




class DocumentManager :
    def __init__(self):
        # OpenAIEmbeddingsの初期化
        self.emb_model = OpenAIEmbeddings()


    #PDFの読み込みと分割をまとめた関数
    def load_and_process_pdfs(self,pdf_files):
        try:
            # 各PDFを読み込んで分割しリストに格納
            split_sections = []
            for pdf_file in pdf_files:
                text_to_split = self.read_pdf(pdf_file["path"])
                split_section = self.split_text_by_keyword(text_to_split, pdf_file["keyword"])
                split_sections.append(split_section)

            return split_sections
        
        except Exception as e:
            st.error("PDF処理中にエラーが発生しました。再度お試しください。")
            return None


    # PDFのファイルを読み込む
    def read_pdf(self,file_path):
        try:
            doc =fitz.open(file_path)
            text = ""
            # PDFの全ページを抽出する
            for page_num in range(len(doc)):
                # テキストを取得する
                text += doc.load_page(page_num).get_text()

            return text
        
        except Exception as e:
            st.error("PDFの読み込みにエラーが発生しました。再度お試しください。")
            return None

    
    #指定したキーワードでテキストを分割し、リストに格納する
    def split_text_by_keyword(self,text_to_split,start_keyword):  
        try:     
            end_keyword = "---"
            #分割条件の設定
            pattern = rf"({re.escape(start_keyword)}.*?)(?={re.escape(end_keyword)}|\Z)"
            #指定したキーワードでテキストを分割する
            chunks = re.findall(pattern,text_to_split,re.DOTALL)

            split_section = []
            for chunk in chunks:
                #\nで分割し、行ごとにリストに格納
                lines = chunk.split("\n")
                #セクション番号を変数に格納
                number = lines[0]
                #残ったリストを1つにまとめる
                content = "\n".join(lines[1:])
                split_section.append({"number": number, "content": content})

            return split_section

        except Exception as e:
            st.error("PDFの分割中にエラーが発生しました。再度お試しください。")
            return None


    #前処理されたセクションをベクトル化する
    def create_vectorstore(self,split_sections):
        try:
            documents = []
            # 各セクションの内容を使ってAIが扱いやすいドキュメント形式に変え、リストに追加する
            for sections in split_sections:
                for section in sections:
                    number = section["number"]
                    content = section["content"]
                    full_text = f"{number} {content}"
                    # full_textを使ってドキュメントオブジェクトを作成し、AIが扱いやすい形式にする
                    documents.append(Document(page_content=full_text))

            #リストで複数のドキュメントを渡してベクトルストアを作成
            vectorstore = FAISS.from_documents(documents,self.emb_model)

            return vectorstore
        
        except Exception as e:
            st.error("入力データの処理中にエラーが発生しました。再度お試しください。")
            return None