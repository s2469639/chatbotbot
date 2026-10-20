import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. 환경 변수 로드 및 API Key 확인
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. 페이지 기본 설정
st.set_page_config(
    page_title="대화 기억 AI 챗봇",
    page_icon="🧠",
    layout="centered"
)

# 3. 사이드바 레이아웃 (설정 및 대화 기록 관리)
with st.sidebar:
    st.header("설정")
    
    # 모델 선택박스 (기본값 gemini-3.8-flash)
    model_name = st.selectbox(
        "모델 선택",
        options=["gemini-3.8-flash", "gemini-3.8-pro"],
        index=0,
        help="사용할 Gemini 모델을 선택하세요."
    )
    
    # temperature 슬라이더
    temperature = st.slider(
        "Temperature (창의성)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="값이 높을수록 창의적이고, 낮을수록 일관된 답변을 제공합니다."
    )
    
    # 기억할 최근 대화 개수 슬라이더 (03memory_readme 기준 최대 50개 등 조정 가능)
    max_history_turns = st.slider(
        "기억할 최근 대화 수 (턴)",
        min_value=2,
        max_value=50,
        value=20,
        step=2,
        help="API로 전송할 최근 대화(사용자+AI 합산) 개수를 제한합니다."
    )
    
    st.divider()
    
    # 대화 통계 및 초기화 버튼
    if "messages" in st.session_state:
        st.write(f"현재 저장된 메시지: **{len(st.session_state.messages)}개**")
        
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. 메인 화면 제목
st.title("🧠 대화 기억 AI 챗봇")
st.caption("이전 대화를 기억하여 문맥에 맞는 자연스러운 답변을 제공합니다.")

# API 키 검증
if not api_key:
    st.error(".env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# 5. Gemini 클라이언트 초기화
@st.cache_resource
def get_client(key: str):
    return genai.Client(api_key=key)

client = get_client(api_key)

# 6. 세션 상태(대화 기록) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 7. 기존 대화 기록 화면에 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 8. 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 8-1. 사용자 메시지 화면 출력 및 세션 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 8-2. 최근 대화 내역(메모리)을 Gemini API Content 형식으로 변환
    # 최근 max_history_turns 개수만큼만 슬라이싱하여 문맥 전달
    recent_messages = st.session_state.messages[-max_history_turns:]
    
    contents_payload = []
    for msg in recent_messages:
        # Gemini API는 role로 'user'와 'model'을 사용 (Streamlit의 assistant는 model로 매핑)
        role = "user" if msg["role"] == "user" else "model"
        contents_payload.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # 8-3. AI 응답 생성 및 스트리밍 출력
    with st.chat_message("assistant"):
        try:
            def response_generator():
                response_stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents_payload,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        thinking_config=types.ThinkingConfig(
                            thinking_level="low"
                        )
                    )
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield chunk.text

            full_response = st.write_stream(response_generator())
            
            if not full_response:
                st.markdown("(응답 내용이 없습니다.)")
                full_response = "(응답 내용이 없습니다.)"

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
