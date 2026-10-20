import os
import chardet
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAL_API_KEY") or os.getenv("OPEN_API_KEY")

# 페이지 레이아웃 및 타이틀 설정
st.set_page_config(
    page_title="무역 실무 전문 RAG 챗봇",
    layout="wide"
)

def read_file_with_encoding(file_path: str) -> str:
    """UTF-8, UTF-8-sig, CP949를 포함하여 파일 인코딩을 자동 감지하고 디코딩하는 함수"""
    with open(file_path, "rb") as f:
        raw_data = f.read()

    # chardet을 통한 인코딩 감지
    detected = chardet.detect(raw_data)
    detected_encoding = detected.get("encoding")

    candidate_encodings = [detected_encoding, "utf-8", "utf-8-sig", "cp949", "euc-kr"]
    
    for enc in candidate_encodings:
        if not enc:
            continue
        try:
            return raw_data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

    return raw_data.decode("utf-8", errors="replace")

def load_trade_guide(file_path: str):
    """trade_guide.txt 문서를 로드하고 RecursiveCharacterTextSplitter로 분할"""
    if not os.path.exists(file_path):
        return []

    text_content = read_file_with_encoding(file_path)
    file_name = os.path.basename(file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

    raw_chunks = text_splitter.split_text(text_content)
    documents = [
        Document(page_content=chunk, metadata={"source": file_name})
        for chunk in raw_chunks
    ]
    return documents

def load_products_csv(file_path: str):
    """sample_products.csv를 행(Row)별 [컬럼명]: [값] 형태의 Document 객체로 생성"""
    if not os.path.exists(file_path):
        return []

    text_content = read_file_with_encoding(file_path)
    file_name = os.path.basename(file_path)

    from io import StringIO
    df = pd.read_csv(StringIO(text_content))

    documents = []
    for _, row in df.iterrows():
        row_parts = []
        for col_name, val in row.items():
            row_parts.append(f"[{col_name}]: {val}")
        row_content = "\n".join(row_parts)

        documents.append(
            Document(page_content=row_content, metadata={"source": file_name})
        )
    return documents

@st.cache_resource(show_spinner="무역 실무 데이터 및 벡터 데이터베이스를 구축하는 중입니다...")
def initialize_retriever():
    """문서 로드, 임베딩 생성 및 Chroma 벡터 DB 캐싱"""
    from langchain_community.vectorstores import Chroma

    # langchain-openai 설치 여부에 따라 fallback 임포트
    try:
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key
        )
    except ImportError:
        from langchain_community.embeddings import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key
        )

    # 데이터 로드
    guide_docs = load_trade_guide(os.path.join("data", "trade_guide.txt"))
    product_docs = load_products_csv(os.path.join("data", "sample_products.csv"))
    all_documents = guide_docs + product_docs

    if not all_documents:
        return None

    # Chroma in-memory 벡터 저장소 생성
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings
    )

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

def format_docs_with_sources(docs):
    """검색된 문서 내용과 메타데이터 소스를 결합"""
    formatted_chunks = []
    for doc in docs:
        source_name = doc.metadata.get("source", "알 수 없음")
        formatted_chunks.append(f"[출처: {source_name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_chunks)

def get_llm():
    """OpenAI Chat LLM 인스턴스 반환"""
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            api_key=api_key
        )
    except ImportError:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage
        from openai import OpenAI
        
        class DirectOpenAIChatModel(BaseChatModel):
            """langchain-openai 패키지 미설치 시 순수 openai 라이브러리로 LCEL 파이프라인 구동"""
            model_name: str = "gpt-4o-mini"
            temperature: float = 0.0

            @property
            def _llm_type(self) -> str:
                return "direct_openai_chat"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                client = OpenAI(api_key=api_key)
                formatted_msgs = []
                for m in messages:
                    role = "user"
                    if m.type == "system":
                        role = "system"
                    elif m.type == "ai":
                        role = "assistant"
                    formatted_msgs.append({"role": role, "content": m.content})

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_msgs,
                    temperature=self.temperature
                )
                text = response.choices[0].message.content or ""
                return {"generations": [{"text": text, "message": AIMessage(content=text)}]}

            def stream(self, input, config=None, **kwargs):
                client = OpenAI(api_key=api_key)
                messages = self._convert_input(input).to_messages()
                formatted_msgs = []
                for m in messages:
                    role = "user"
                    if m.type == "system":
                        role = "system"
                    elif m.type == "ai":
                        role = "assistant"
                    formatted_msgs.append({"role": role, "content": m.content})

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_msgs,
                    temperature=self.temperature,
                    stream=True
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

        return DirectOpenAIChatModel()

# 사이드바 레이아웃
with st.sidebar:
    st.header("시스템 설정")
    st.markdown("**기본 모델:** OpenAI gpt-4o-mini")
    st.markdown("**임베딩 모델:** text-embedding-3-small")
    st.markdown("**벡터 데이터베이스:** Chroma")
    st.markdown("**참조 데이터:**\n- trade_guide.txt\n- sample_products.csv")

    st.divider()
    if st.button("대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 메인 UI
st.title("무역 실무 전문 질의응답 시스템")
st.caption("20년 경력의 무역 전문가가 사내 무역 가이드 및 제품 데이터를 기반으로 정확한 답변을 제공합니다.")

if not api_key:
    st.error(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# Retriever 초기화
retriever = initialize_retriever()
if retriever is None:
    st.error("데이터 파일을 찾을 수 없거나 로드에 실패했습니다. (data/trade_guide.txt, data/sample_products.csv)")
    st.stop()

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# LCEL 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_template(
    """당신은 20년 경력의 노련하고 신뢰감 있는 무역 전문가입니다.
인코텀즈(Incoterms), HS Code, 선하증권(B/L), 수출입 통관 절차 등 전문 용어를 정확하고 명료하게 설명하는 어조로 답변하십시오.

[참고 자료]
{context}

[질문]
{question}

[답변 작성 시 엄격한 제약사항]
1. 반드시 위에 제공된 [참고 자료]의 내용에만 근거하여 답변하십시오.
2. 질문 내용이 [참고 자료]에 전혀 나와 있지 않거나 확인할 수 없는 경우, 절대 외부 지식이나 추측을 덧붙이지 말고 정확히 "자료에 없습니다."라고만 답변하십시오.
3. 답변 본문 하단에 반드시 정보의 근거가 된 출처 파일명을 아래 형식으로 명확히 표시하십시오.
   (형식: 참고 출처: 파일명1, 파일명2)
   단, 답변 내용이 "자료에 없습니다."인 경우에는 출처를 표시하지 마십시오.
4. 모든 문장에서 이모지(이모티콘, 특수 그림문자)를 절대로 사용하지 마십시오.
"""
)

# LCEL 체인 구성 (Retriever | Prompt | LLM | StrOutputParser)
llm = get_llm()

# 질문 입력 처리
if user_input := st.chat_input("무역 실무 또는 제품 정보에 대해 질문을 입력하십시오."):
    # 사용자 질문 렌더링 및 저장
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # AI 응답 생성 및 스트리밍
    with st.chat_message("assistant"):
        try:
            # 1. 관련 문서 검색
            retrieved_docs = retriever.invoke(user_input)
            context_text = format_docs_with_sources(retrieved_docs)

            # 2. LCEL 체인 생성
            rag_chain = prompt_template | llm | StrOutputParser()

            # 3. 토큰 실시간 스트리밍
            def stream_response():
                for chunk in rag_chain.stream({"context": context_text, "question": user_input}):
                    yield chunk

            full_response = st.write_stream(stream_response())

            if not full_response:
                full_response = "자료에 없습니다."
                st.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
