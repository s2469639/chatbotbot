import os
import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 환경 변수 로드 및 API Key 확인
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. 페이지 기본 설정
st.set_page_config(
    page_title="LangChain 2문장 AI 챗봇",
    page_icon="🦜🔗",
    layout="centered"
)

# 3. 사이드바 레이아웃
with st.sidebar:
    st.header("⚙️ LangChain 설정")
    
    # 모델 선택박스
    model_name = st.selectbox(
        "모델 선택",
        options=["gemini-3.8-flash", "gemini-3.8-pro", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="사용할 LangChain ChatGoogleGenerativeAI 모델을 선택하세요."
    )
    
    # temperature 슬라이더
    temperature = st.slider(
        "Temperature (창의성)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1
    )
    
    # 최근 대화 기억 개수 슬라이더
    max_history_turns = st.slider(
        "기억할 최근 대화 수 (턴)",
        min_value=2,
        max_value=50,
        value=20,
        step=2,
        help="LangChain 프롬프트로 전송할 최근 대화 메시지 수입니다."
    )
    
    st.divider()
    
    # 대화 통계 및 초기화 버튼
    if "messages" in st.session_state:
        st.write(f"현재 대화: **{len(st.session_state.messages)}개 메시지**")

    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. 메인 화면 제목
st.title("🦜🔗 LangChain 챗봇")
st.caption("LangChain의 LCEL(Prompt | LLM | Parser) 체인을 사용하며, 모든 답변은 정확히 2문장으로 제공됩니다.")

# API 키 검증
if not api_key:
    st.error(".env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# 5. 세션 상태(대화 기록) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 6. 기존 대화 기록 화면에 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 7. LangChain 체인 구성 함수
def get_chain(model: str, temp: float, key: str):
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temp,
        google_api_key=key
    )
    
    # 2문장 제약 조건을 포함한 시스템 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "당신은 유능하고 친절한 AI 어시스턴트입니다.\n"
            "[중요 규칙]\n"
            "1. 반드시 정확히 '2문장'으로만 답변해야 합니다.\n"
            "2. 너무 길거나 짧지 않게 핵심 내용을 명확히 담아 2문장으로 완성하세요.\n"
            "3. 한국어로 자연스럽고 정중하게 답변하세요."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    # LCEL (LangChain Expression Language) 파이프라인 조립
    return prompt | llm | StrOutputParser()

# 8. 사용자 입력 처리
if user_input := st.chat_input("메시지를 입력하세요..."):
    # 8-1. 사용자 메시지 화면 출력 및 세션 저장
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 8-2. 세션 기록을 LangChain 메시지 객체(HumanMessage, AIMessage)로 변환 (최근 N개 적용)
    history_messages = []
    # 방금 추가된 최신 user 메시지를 제외한 직전 대화들을 history로 전달
    recent_history = st.session_state.messages[:-1][-max_history_turns:]
    for msg in recent_history:
        if msg["role"] == "user":
            history_messages.append(HumanMessage(content=msg["content"]))
        else:
            history_messages.append(AIMessage(content=msg["content"]))

    # 8-3. LangChain 스트리밍 호출 및 출력
    with st.chat_message("assistant"):
        try:
            chain = get_chain(model_name, temperature, api_key)

            # chain.stream을 제너레이터로 감싸 st.write_stream에 전달
            def stream_response():
                for chunk in chain.stream({
                    "history": history_messages,
                    "input": user_input
                }):
                    yield chunk

            full_response = st.write_stream(stream_response())

            if not full_response:
                st.markdown("(응답 내용이 없습니다.)")
                full_response = "(응답 내용이 없습니다.)"

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
