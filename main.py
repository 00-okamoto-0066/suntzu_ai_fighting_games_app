import streamlit as st
import yaml
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from conversation_management import ConversationManagement
from model_setup import ModelSetup
from document_manager import DocumentManager 
from langchain_openai import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    AIMessage
)


#Streamlitのページ設定やヘッダーやサイドバーのオプション表示。
st.set_page_config(
page_title="格闘ゲームをサポートする孫子AI",
page_icon="🤖"
)
st.header("格闘ゲームをサポートする孫子AI 🤖")
st.sidebar.title("Options")



class State (TypedDict):
    messages: Annotated[list,add_messages]


@tool
def suntzu_ai_tool(question):
    """悩んでいるユーザーを解決する孫子API"""
    vectorstore = st.session_state.vectorstore
    answer = create_suntzu_qa_chain(question,vectorstore)
    return answer



# 指定された言語モデル（LLM）とベクターストアを用いて質問応答システムを作成する関数
def create_suntzu_qa_chain(question,vectorstore):
    try:
        # sessionから会話履歴とプロンプトを取得
        memory = st.session_state.memory
        qa_prompt = st.session_state.qa_prompt
        comparison_prompt = st.session_state.comparison_prompt    
        messages = memory.load_memory_variables({})

        # mmrを使用し、最も多様性で関連性の高い3つの結果を返すように設定。
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k":3}
        )

        # 会話履歴とベクトルストアを使ったretrieverを作成
        suntzu_answer_retriever = create_history_aware_retriever(llm,retriever,qa_prompt)
        # 文書から回答を生成するチェーンを作成
        comparison_answer_chain = create_stuff_documents_chain(llm,comparison_prompt)
        # 検索と回答生成を統合したretrieval_chainを作成
        chain = create_retrieval_chain(suntzu_answer_retriever, comparison_answer_chain)
        # chainを実行し、質問と会話履歴に基づいて回答を生成
        result = chain.invoke({"input": question, "chat_history": messages["history"]})

        return result["answer"]
    
    except Exception as e:
        st.error("回答の作成中にエラーが発生しました。再度お試しください。")
        return None



#メッセージを追加して状態を更新する関数
def chatbot(state):
    state["messages"].append(llm_with_tool.invoke(state["messages"]))
    return state

# ツールを実行するための関数
def tool(state):
    try:
        tool_by_name = {"suntzu_ai_tool":suntzu_ai_tool}
        last_message = state["messages"][-1]

        # 最新メッセージの詳細情報を使ってツールを呼び出す
        tool_function = tool_by_name[last_message.tool_calls[0]["name"]]
        # ツールを実行して結果を取得
        tool_output = tool_function.invoke(last_message.tool_calls[0]["args"])
        state["messages"].append(ToolMessage(content = tool_output ,tool_call_id = last_message.tool_calls[0]["id"]))

        return state
    except Exception as e:
        st.error("ツールを実行中にエラーが発生しました。再度お試しください。")
        return None


# グラフ構造の作成とコンパイル
graph = StateGraph(State)

# ノードを追加
graph.add_node("chatbot",chatbot)
graph.add_node("tool",tool)

# エントリーポイントとフィニッシュポイントの設定
graph.set_entry_point("chatbot")
graph.set_finish_point("tool")

# エッジを追加
graph.add_edge("chatbot","tool")

runner = graph.compile()




# ユーザーのプロンプトに応じて回答を取得する関数
def get_response(question,runner):
    with st.spinner("ChatGPT is typing ..."):
        response = runner.invoke({"messages":[question]})
    return response["messages"][-1].content


#　クラスのインスタンス化
cm = ConversationManagement()
ms = ModelSetup()
dm = DocumentManager()



#会話履歴を処理する
cm.init_messages()

#コンテナを作成
container = st.container()

#モデルとランクの選択
llm = ChatOpenAI(model_name ="gpt-4o-mini-2024-07-18")
rank = ms.select_rank()

# LLMにツールを設定
llm_with_tool = llm.bind_tools([suntzu_ai_tool])


# YAMLファイルを読み込む 
if"pdf_files" not in st.session_state:
    with open("config.yaml","r") as file: 
        config = yaml.safe_load(file)

    # pdf_filesリストを取得 
    st.session_state.pdf_files = config["pdf_files"]

# PDFの読み込みと分割を行う
if "split_sections" not in st.session_state :  
    pdf_files = st.session_state.pdf_files
    st.session_state.split_sections = dm.load_and_process_pdfs(pdf_files)
    

# 分割した内容をベクトル化する
if "vectorstore" not in st.session_state:
    split_sections = st.session_state.split_sections
    st.session_state.vectorstore = dm.create_vectorstore(split_sections)


if question := st.chat_input("聞きたいことを入力してね！"):
    # ランクごとに対応したプロンプトを作成 
    st.session_state.qa_prompt = cm.generate_prompt(rank)
    st.session_state.comparison_prompt = cm.generate_comparison_prompt(rank)
    results = get_response(question,runner)
else:
    results = None


if results:
    memory = st.session_state.memory
    memory.save_context({"input": question}, {"output": results})
    messages = memory.load_memory_variables({})
    with container:
        for result in messages["history"]:
            if isinstance(result, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(result.content)
            elif isinstance(result, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(result.content)


