import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

# .env 파일 로드
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("오류: .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    sys.exit(1)

# Gemini 클라이언트 초기화
client = genai.Client(api_key=api_key)
model_name = "gemini-3.8-flash"

print("=" * 50)
print("01. 기본 Gemini 챗봇 (종료하려면 'exit' 또는 'quit' 입력)")
print("=" * 50)

while True:
    try:
        user_input = input("당신: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break

        # 01basic_readme.md 사양: 단일 턴 질의응답 (temperature=0.7)
        response = client.models.generate_content(
            model=model_name,
            contents=user_input,
            config=types.GenerateContentConfig(
                temperature=0.7,
            ),
        )

        print(f"AI: {response.text}\n")

    except KeyboardInterrupt:
        print("\n챗봇을 종료합니다.")
        break
    except Exception as e:
        print(f"오류가 발생했습니다: {e}\n")
