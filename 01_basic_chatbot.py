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
print(f"01. 기본 Gemini 챗봇 ({model_name})")
print("종료하려면 'exit', 'quit', 또는 '종료'를 입력하세요.")
print("=" * 50)

while True:
    try:
        user_input = input("당신: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "종료"]:
            print("챗봇을 종료합니다.")
            break

        print("AI: ", end="", flush=True)

        # 스트리밍 방식(generate_content_stream)으로 실시간 출력
        # Gemini 3.8 Flash는 thinking_config(level="low")로 빠른 응답 유도
        response_stream = client.models.generate_content_stream(
            model=model_name,
            contents=user_input,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_level="low"
                )
            ),
        )

        has_output = False
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                has_output = True

        if not has_output:
            print("(응답 내용이 없습니다.)", end="")
        print("\n")

    except KeyboardInterrupt:
        print("\n챗봇을 종료합니다.")
        break
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}\n")
