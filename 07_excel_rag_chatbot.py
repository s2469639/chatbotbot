import os
import io
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 1. 환경 변수 로드 (.env 직접 읽지 않고 os.getenv로만 참조)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAL_API_KEY") or os.getenv("OPEN_API_KEY")

# 2. 페이지 설정 (2열 와이드 레이아웃)
st.set_page_config(
    page_title="Excel/CSV + 문서 RAG 챗봇 (OpenAI)",
    page_icon="📊",
    layout="wide"
)

# 3. 데이터 로드 및 변환 헬퍼 함수
def read_csv_safe(file_or_path):
    """utf-8-sig, utf-8, cp949, euc-kr 인코딩 순차 시도로 한글 깨짐 방지"""
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]
    if hasattr(file_or_path, "read"):
        content = file_or_path.read()
        for enc in encodings:
            try:
                return pd.read_csv(io.BytesIO(content), encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception:
                break
    else:
        for enc in encodings:
            try:
                return pd.read_csv(file_or_path, encoding=enc)
            except UnicodeDecodeError:
                continue
            except Exception:
                break
    raise ValueError("CSV 파일 인코딩(utf-8, cp949 등)을 인식하지 못했습니다.")

def read_txt_safe(file_or_path):
    """텍스트 문서 인코딩 안전 로드"""
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]
    if hasattr(file_or_path, "read"):
        content = file_or_path.read()
        for enc in encodings:
            try:
                return content.decode(enc)
            except UnicodeDecodeError:
                continue
    else:
        if os.path.exists(file_or_path):
            with open(file_or_path, "rb") as f:
                content = f.read()
            for enc in encodings:
                try:
                    return content.decode(enc)
                except UnicodeDecodeError:
                    continue
    return ""

def df_to_text(df: pd.DataFrame) -> str:
    """tabulate 미설치 시에도 에러 없이 텍스트 변환"""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)

# 4. 숫자 계산 질문 판단 및 pandas 연산 실행 함수
def execute_pandas_calculation(client, model: str, df: pd.DataFrame, question: str):
    """
    질문에 집계/합계/평균/최대/최소/개수 등 계산이 필요한 경우
    GPT에게 실행 가능한 pandas 코드를 작성하게 한 후 df를 기반으로 정확히 계산
    """
    calc_keywords = [
        "얼마", "합계", "총", "평균", "개수", "몇 개", "몇개", "최대", "최소", 
        "가장 비싼", "가장 싼", "비율", "더하면", "빼면", "곱하면", "차이", "통계", "수량"
    ]
    needs_calc = any(kw in question for kw in calc_keywords)
    if not needs_calc:
        return None, None

    col_info = []
    for col in df.columns:
        col_info.append(f"- {col} (타입: {df[col].dtype})")
    cols_desc = "\n".join(col_info)

    head_sample = df.head(3).to_string()

    code_gen_prompt = (
        f"당신은 Pandas 데이터 분석 전문가입니다. 주어진 DataFrame `df`를 바탕으로 사용자의 질문에 답하기 위한 Python 계산 코드를 작성하세요.\n\n"
        f"[DataFrame 정보]\n{cols_desc}\n\n"
        f"[데이터 샘플 (상위 3행)]\n{head_sample}\n\n"
        f"[질문]\n{question}\n\n"
        f"[작성 규칙]\n"
        f"1. 반드시 계산 결과값을 `result`라는 변수에 할당하세요.\n"
        f"2. 만약 질문 내용이 df 컬럼이나 데이터와 전혀 관련이 없거나 계산할 수 없으면 `result = None`으로 작성하세요.\n"
        f"3. 오직 순수 Python 코드만 출력하세요. (마크다운 ```python 태그나 설명문 일체 금지)"
    )

    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": code_gen_prompt}],
            temperature=0.0
        )
        code = res.choices[0].message.content.strip()
        code = re.sub(r"^```python\s*", "", code)
        code = re.sub(r"^```\s*", "", code)
        code = re.sub(r"\s*```$", "", code).strip()

        # 안전한 실행 환경 구성
        local_vars = {"df": df.copy(), "pd": pd}
        exec(code, {}, local_vars)
        result = local_vars.get("result")

        if result is not None:
            return str(result), code
    except Exception:
        return None, None

    return None, None

# 5. 세션 상태 초기화
if "table_df" not in st.session_state:
    st.session_state.table_df = None
    st.session_state.table_name = ""

if "guide_text" not in st.session_state:
    st.session_state.guide_text = ""
    st.session_state.guide_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# 기본 자료 자동 로드 (data/sample_products.csv, data/trade_guide.txt)
if st.session_state.table_df is None:
    default_csv_path = os.path.join("data", "sample_products.csv")
    if os.path.exists(default_csv_path):
        try:
            st.session_state.table_df = read_csv_safe(default_csv_path)
            st.session_state.table_name = "sample_products.csv (기본)"
        except Exception:
            pass

if not st.session_state.guide_text:
    for txt_candidate in [os.path.join("data", "trade_guide.txt"), os.path.join("data", "trade.txt")]:
        if os.path.exists(txt_candidate):
            text_loaded = read_txt_safe(txt_candidate)
            if text_loaded:
                st.session_state.guide_text = text_loaded
                st.session_state.guide_name = f"{os.path.basename(txt_candidate)} (기본)"
                break

# 6. 사이드바 (설정 & 파일 관리)
with st.sidebar:
    st.header("📂 데이터 파일 관리")
    
    uploaded_table = st.file_uploader(
        "Excel 또는 CSV 파일 업로드",
        type=["csv", "xlsx", "xls"],
        help="한글 CSV (CP949, UTF-8) 및 Excel 파일을 지원합니다."
    )
    if uploaded_table is not None:
        try:
            if uploaded_table.name.endswith(".csv"):
                df = read_csv_safe(uploaded_table)
            else:
                df = pd.read_excel(uploaded_table)
            st.session_state.table_df = df
            st.session_state.table_name = uploaded_table.name
            st.success(f"'{uploaded_table.name}' 적용 완료 ({len(df)}행)")
        except Exception as e:
            st.error(f"테이블 파일 로드 실패: {e}")

    uploaded_text = st.file_uploader(
        "가이드 문서 업로드 (TXT)",
        type=["txt", "md"],
        help="추가 참조할 가이드 문서입니다."
    )
    if uploaded_text is not None:
        try:
            txt_content = read_txt_safe(uploaded_text)
            st.session_state.guide_text = txt_content
            st.session_state.guide_name = uploaded_text.name
            st.success(f"'{uploaded_text.name}' 적용 완료")
        except Exception as e:
            st.error(f"문서 로드 실패: {e}")

    if st.button("🔄 기본 샘플 데이터로 복원", use_container_width=True):
        default_csv_path = os.path.join("data", "sample_products.csv")
        if os.path.exists(default_csv_path):
            st.session_state.table_df = read_csv_safe(default_csv_path)
            st.session_state.table_name = "sample_products.csv (기본)"
        default_txt_path = os.path.join("data", "trade_guide.txt")
        if os.path.exists(default_txt_path):
            st.session_state.guide_text = read_txt_safe(default_txt_path)
            st.session_state.guide_name = "trade_guide.txt (기본)"
        st.rerun()

    st.divider()

    st.header("⚙️ 모델 설정 (OpenAI)")
    model_name = st.selectbox(
        "모델 선택",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
        help="기본으로 gpt-4o-mini를 사용합니다."
    )

    temperature = st.slider(
        "Temperature (정확도 우선)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="정확한 데이터 기반 답변을 위해 0.0을 권장합니다."
    )

    max_history_turns = st.slider(
        "기억할 최근 대화 수",
        min_value=2,
        max_value=20,
        value=10,
        step=2
    )

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 7. 메인 화면 제목
st.title("📊 Excel/CSV + 문서 RAG 챗봇 (OpenAI)")
st.caption("질문에 계산이 필요하면 Pandas 코드를 직접 실행하여 정확한 수치를 도출하고, 없는 내용은 '데이터에 없습니다.'라고 답변합니다.")

if not api_key:
    st.error(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

@st.cache_resource
def get_openai_client(key: str):
    return OpenAI(api_key=key)

client = get_openai_client(api_key)

# 8. 메인 2열 레이아웃 구성
col_data, col_chat = st.columns([1, 1], gap="medium")

with col_data:
    st.subheader("📑 로드된 자료 미리보기")

    with st.expander(f"📊 표 데이터: {st.session_state.table_name or '없음'}", expanded=True):
        if st.session_state.table_df is not None:
            df = st.session_state.table_df
            st.caption(f"총 {len(df)}개 행 / 컬럼: {', '.join(df.columns.astype(str))}")
            st.dataframe(df, use_container_width=True, height=260)
        else:
            st.info("로드된 Excel/CSV 데이터가 없습니다.")

    with st.expander(f"📖 텍스트 문서: {st.session_state.guide_name or '없음'}", expanded=True):
        if st.session_state.guide_text:
            st.text_area(
                "문서 내용 미리보기",
                value=st.session_state.guide_text,
                height=260,
                disabled=True
            )
        else:
            st.info("로드된 텍스트 가이드 문서가 없습니다.")

with col_chat:
    st.subheader("💬 데이터 기반 Q&A 채팅")

    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_question := st.chat_input("질문을 입력하세요 (예: 재고수량의 총합은 얼마인가요?)"):
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # 8-1. 숫자 계산 필요 여부 확인 및 Pandas 직접 계산 실행
        calculated_result = None
        calc_code = None
        if st.session_state.table_df is not None:
            with st.spinner("데이터 분석 및 수치 계산 중..."):
                calculated_result, calc_code = execute_pandas_calculation(
                    client=client,
                    model=model_name,
                    df=st.session_state.table_df,
                    question=user_question
                )

        # 8-2. 프롬프트 컨텍스트 구성
        table_context_str = df_to_text(st.session_state.table_df) if st.session_state.table_df is not None else "(제공된 표 데이터 없음)"
        guide_context_str = st.session_state.guide_text if st.session_state.guide_text else "(제공된 가이드 문서 없음)"
        table_label = st.session_state.table_name or "없음"
        guide_label = st.session_state.guide_name or "없음"

        calc_context = ""
        if calculated_result is not None:
            calc_context = f"\n[Pandas 실행 계산 결과]\n실행 코드:\n```python\n{calc_code}\n```\n계산 결과값: {calculated_result}\n"

        system_prompt = (
            "당신은 주어진 데이터(표 데이터 및 무역 가이드 문서)에만 엄격히 근거하여 답변하는 AI 어시스턴트입니다.\n\n"
            f"[제공된 컨텍스트 데이터]\n"
            f"--- [표 데이터 (Excel/CSV: {table_label})] ---\n"
            f"{table_context_str}\n\n"
            f"--- [참조 가이드 문서 ({guide_label})] ---\n"
            f"{guide_context_str}\n"
            f"{calc_context}\n"
            "[답변 원칙 - 반드시 준수]\n"
            "1. 반드시 제공된 [표 데이터] 또는 [참조 가이드 문서] 내용에만 근거하여 답변하세요.\n"
            "2. 'FOB와 CIF 차이', '인코텀즈', '수출 절차', 'HS 코드', '필수 서류' 등 무역 관련 질문은 [참조 가이드 문서]의 내용을 바탕으로 상세하고 정확하게 설명하세요.\n"
            "3. 제품 가격, 재고, 코드, 통계 등 정형 데이터 관련 질문은 [표 데이터] 및 계산 결과를 바탕으로 답변하세요.\n"
            "4. [표 데이터]와 [참조 가이드 문서] 어디에도 없는 내용은 절대 임의로 지어내거나 추측하지 말고 반드시 \"자료에 없습니다.\"라고만 답변하세요.\n"
            "5. 답변의 맨 마지막 줄에는 반드시 정보를 찾은 출처 파일명을 아래 형식으로 표기하세요:\n"
            "   - 가이드 문서에서 찾은 경우: 📌 출처: trade_guide.txt\n"
            "   - 엑셀/CSV에서 찾은 경우: 📌 출처: sample_products.csv\n"
            "   - 둘 다 참조한 경우: 📌 출처: sample_products.csv, trade_guide.txt\n"
            "   - 단, \"자료에 없습니다.\"라고 답변할 때는 출처를 표기하지 않습니다.\n"
            "6. 한국어로 정중하고 명확하게 답변하세요."
        )

        api_messages = [{"role": "system", "content": system_prompt}]
        for msg in st.session_state.messages[:-1][-max_history_turns:]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
        api_messages.append({"role": "user", "content": user_question})

        # 8-3. 스트리밍 답변 생성
        with chat_container:
            with st.chat_message("assistant"):
                try:
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=api_messages,
                        temperature=temperature,
                        stream=True
                    )

                    def response_stream():
                        for chunk in stream:
                            text = chunk.choices[0].delta.content
                            if text:
                                yield text

                    full_response = st.write_stream(response_stream())

                    if not full_response:
                        full_response = "자료에 없습니다."
                        st.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
