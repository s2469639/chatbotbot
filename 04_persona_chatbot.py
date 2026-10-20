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
    page_title="페르소나 시스템 프롬프트 챗봇",
    page_icon="🎭",
    layout="centered"
)

# 3. 기본 페르소나 템플릿 정의
PERSONA_TEMPLATES = {
    "무역 전문가": (
        "당신은 20년 경력의 무역 전문가입니다.\n"
        "- 수출입 절차, 관세, 인코텀즈(Incoterms)에 대해 전문적이고 실무적으로 답변합니다.\n"
        "- 복잡한 내용도 20년 현장 실무 경험을 바탕으로 이해하기 쉽게 설명합니다.\n"
        "- 필요한 관련 법규 및 필수 서류 양식도 친절하게 안내합니다.\n"
        "- 항상 정중하고 전문적인 어조의 한국어로 답변합니다."
    ),
    "코딩 튜터": (
        "당신은 Python 프로그래밍 20년차 친절한 코딩 튜터입니다.\n"
        "- 초보자도 쉽게 이해할 수 있도록 눈높이에 맞춰 친절하게 설명합니다.\n"
        "- 코드 예제에는 반드시 친절한 주석을 달아 설명합니다.\n"
        "- 코드의 실행 결과와 동작 원리도 함께 안내합니다.\n"
        "- 오류가 있는 코드는 원인을 짚어준 뒤 수정된 코드를 제시합니다."
    ),
    "직접 입력": ""
}

# 4. 사이드바 레이아웃 (페르소나 및 파라미터 설정)
with st.sidebar:
    st.header("🎭 페르소나 설정")

    # 페르소나 선택
    selected_persona = st.selectbox(
        "페르소나 선택",
        options=list(PERSONA_TEMPLATES.keys()),
        index=0,
        help="원하는 AI 역할을 선택하세요."
    )

    # 선택 변경 시 텍스트 영역 기본값 갱신을 위한 세션 상태 관리
    if "current_persona" not in st.session_state:
        st.session_state.current_persona = selected_persona
        st.session_state.system_prompt = PERSONA_TEMPLATES[selected_persona]

    if st.session_state.current_persona != selected_persona:
        st.session_state.current_persona = selected_persona
        st.session_state.system_prompt = PERSONA_TEMPLATES[selected_persona]

    # 시스템 프롬프트 직접 수정 가능한 텍스트 영역
    system_instruction = st.text_area(
        "시스템 프롬프트 (직접 수정 가능)",
        value=st.session_state.system_prompt,
        height=180,
        help="AI의 역할, 어조, 답변 규칙 등을 자유롭게 수정할 수 있습니다."
    )
    st.session_state.system_prompt = system_instruction

    st.divider()
    st.header("⚙️ 모델 설정")

    # 모델 선택박스 (기본값 gemini-3.8-flash)
    model_name = st.selectbox(
        "모델 선택",
        options=["gemini-3.8-flash", "gemini-3.8-pro"],
        index=0
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
        step=2
    )

    st.divider()

    # 대화 통계 및 초기화 버튼
    if "messages" in st.session_state:
        st.write(f"현재 대화: **{len(st.session_state.messages)}개 메시지**")

    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. 메인 화면 제목 및 안내
st.title("🎭 페르소나 챗봇")
st.caption(f"현재 역할: **{selected_persona}** (사이드바에서 지시사항을 직접 수정할 수 있습니다)")

# API 키 검증
if not api_key:
    st.error(".env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# 6. Gemini 클라이언트 초기화
@st.cache_resource
def get_client(key: str):
    return genai.Client(api_key=key)

client = get_client(api_key)

# 7. 세션 상태(대화 기록) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 8. 기존 대화 기록 화면에 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 9. 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 9-1. 사용자 메시지 화면 출력 및 세션 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 9-2. 이전 대화 기록을 Gemini API Content 형식으로 변환 (최근 N개 반영)
    recent_messages = st.session_state.messages[-max_history_turns:]
    contents_payload = []
    for msg in recent_messages:
        role = "user" if msg["role"] == "user" else "model"
        contents_payload.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    # 9-3. AI 응답 생성 (시스템 지시사항 적용 및 스트리밍 출력)
    with st.chat_message("assistant"):
        try:
            # GenerateContentConfig의 system_instruction에 사용자가 설정한 프롬프트 전달
            config_kwargs = {
                "temperature": temperature,
                "thinking_config": types.ThinkingConfig(
                    thinking_level="low"
                )
            }
            if system_instruction.strip():
                config_kwargs["system_instruction"] = system_instruction.strip()

            def response_generator():
                response_stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents_payload,
                    config=types.GenerateContentConfig(**config_kwargs)
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
