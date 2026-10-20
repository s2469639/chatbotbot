import os
import sqlite3
import uuid
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# 1. 환경 변수 로드 및 OpenAI API Key 확인
load_dotenv()
# .env의 다양한 키 변수명 대응 (OPENAI_API_KEY, OPENAL_API_KEY, OPEN_API_KEY)
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAL_API_KEY") or os.getenv("OPEN_API_KEY")

DB_FILE = "chat_history.db"

# 2. SQLite 데이터베이스 초기화 및 헬퍼 함수
def init_db():
    """데이터베이스 테이블 생성"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # 세션(대화방) 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT,
                created_at TIMESTAMP
            )
        """)
        # 메시지 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)
        conn.commit()

def get_sessions():
    """모든 세션 목록 조회 (최신순)"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC")
        return cursor.fetchall()

def create_new_session(title="새 대화"):
    """새로운 대화 세션 생성"""
    new_id = str(uuid.uuid4())[:8]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (session_id, title, created_at) VALUES (?, ?, ?)",
            (new_id, title, now)
        )
        conn.commit()
    return new_id

def save_message(session_id, role, content):
    """메시지 저장"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now)
        )
        # 첫 사용자 메시지일 경우 세션 타이틀 업데이트
        if role == "user":
            cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
            count = cursor.fetchone()[0]
            if count <= 2:
                short_title = content[:15] + "..." if len(content) > 15 else content
                cursor.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (short_title, session_id)
                )
        conn.commit()

def load_messages(session_id):
    """특정 세션의 메시지 불러오기"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

def delete_session(session_id):
    """세션 및 해당 대화 내역 삭제"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

# DB 테이블 초기화 실행
init_db()

# 3. 페이지 기본 설정
st.set_page_config(
    page_title="DB 저장 페르소나 챗봇 (OpenAI)",
    page_icon="💾",
    layout="centered"
)

# 4. 페르소나 템플릿 정의 (04시스템프롬프트.md 기반)
PERSONA_TEMPLATES = {
    "기본 비서": (
        "당신은 유능하고 친절한 AI 어시스턴트입니다.\n"
        "- 핵심만 명확하고 간결하게 답변합니다.\n"
        "- 불필요한 미사여구를 줄이고 한국어로 정중하게 답합니다."
    ),
    "무역 전문가": (
        "당신은 20년 경력의 무역 전문가입니다.\n"
        "- 수출입 절차, 관세, 인코텀즈에 대해 전문적으로 답변합니다.\n"
        "- 핵심 실무 요점 위주로 간결하고 명확하게 답변합니다."
    ),
    "코딩 튜터": (
        "당신은 Python 프로그래밍 20년차 튜터입니다.\n"
        "- 초보자도 이해하기 쉽게 핵심 위주로 명확하고 간결하게 답변합니다.\n"
        "- 필요한 코드 조각과 주석만 깔끔하게 제공합니다."
    ),
    "직접 입력": ""
}

# 5. 세션 관리 상태 초기화
sessions = get_sessions()
if not sessions:
    new_sess_id = create_new_session("새 대화")
    st.session_state.current_session_id = new_sess_id
    sessions = get_sessions()
elif "current_session_id" not in st.session_state:
    st.session_state.current_session_id = sessions[0][0]

# 6. 사이드바 (DB 세션 관리 + 페르소나 설정)
with st.sidebar:
    st.header("💾 대화 세션 (DB 관리)")

    # 새 대화 시작 버튼
    if st.button("➕ 새 대화방 만들기", use_container_width=True):
        new_sess_id = create_new_session()
        st.session_state.current_session_id = new_sess_id
        st.rerun()

    # DB에 저장된 세션 목록 선택
    session_dict = {f"{s[1]} ({s[2][:10]})": s[0] for s in sessions}
    session_names = list(session_dict.keys())
    
    current_index = 0
    for idx, sid in enumerate(session_dict.values()):
        if sid == st.session_state.current_session_id:
            current_index = idx
            break

    selected_session_name = st.selectbox(
        "대화 기록 불러오기",
        options=session_names,
        index=current_index,
        help="데이터베이스(SQLite)에 영구 저장된 이전 대화방을 선택합니다."
    )
    st.session_state.current_session_id = session_dict[selected_session_name]

    if st.button("🗑️ 현재 대화방 삭제", use_container_width=True):
        delete_session(st.session_state.current_session_id)
        st.session_state.pop("current_session_id", None)
        st.rerun()

    st.divider()

    st.header("🎭 페르소나 설정")
    selected_persona = st.selectbox(
        "페르소나 선택",
        options=list(PERSONA_TEMPLATES.keys()),
        index=0
    )

    if "current_persona" not in st.session_state or st.session_state.current_persona != selected_persona:
        st.session_state.current_persona = selected_persona
        st.session_state.system_prompt = PERSONA_TEMPLATES[selected_persona]

    system_instruction = st.text_area(
        "시스템 프롬프트 (수정 가능)",
        value=st.session_state.system_prompt,
        height=140
    )
    st.session_state.system_prompt = system_instruction

    st.divider()
    st.header("⚙️ OpenAI 모델 설정")
    model_name = st.selectbox(
        "모델 선택",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

# 7. 메인 화면 제목
st.title("💾 DB 영구 저장 챗봇")
st.caption(f"DB 파일: `{DB_FILE}` (SQLite) | 역할: **{selected_persona}** | 답변 스타일: **간결함**")

# API Key 검증
if not api_key:
    st.error(".env 파일에 OpenAI API Key(OPENAI_API_KEY 또는 OPENAL_API_KEY)가 설정되어 있지 않습니다.")
    st.stop()

# OpenAI 클라이언트 초기화
@st.cache_resource
def get_openai_client(key: str):
    return OpenAI(api_key=key)

client = get_openai_client(api_key)

# 8. DB에서 현재 세션의 대화 내역 불러오기
current_session_id = st.session_state.current_session_id
messages = load_messages(current_session_id)

# 9. 기존 대화 기록 화면에 표시
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 10. 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 10-1. 사용자 메시지 화면 출력 및 DB에 저장
    with st.chat_message("user"):
        st.markdown(prompt)
    save_message(current_session_id, "user", prompt)

    # 10-2. OpenAI API 요청 메시지 구성 (시스템 프롬프트 + 간결한 답변 지시 + 최근 대화 목록)
    concise_rule = "\n[중요 규칙] 답변은 군더더기 없이 간결하고 핵심 위주로 짧게 답변하세요."
    final_system_instruction = (system_instruction + concise_rule).strip()

    api_messages = [{"role": "system", "content": final_system_instruction}]
    
    # DB에 저장된 최근 대화 목록 불러와 추가
    updated_history = load_messages(current_session_id)
    # 최근 20개 메시지만 전달
    for msg in updated_history[-20:]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # 10-3. AI 응답 생성 및 스트리밍 출력
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                temperature=temperature,
                stream=True
            )

            def response_generator():
                for chunk in stream:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            full_response = st.write_stream(response_generator())

            if not full_response:
                full_response = "(응답 내용이 없습니다.)"

            # 10-4. AI 답변 DB에 저장
            save_message(current_session_id, "assistant", full_response)
            
            # 첫 질문 시 사이드바 제목 갱신 등을 위해 새로고침
            if len(updated_history) <= 2:
                st.rerun()

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
