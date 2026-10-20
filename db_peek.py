import sqlite3
import os

DB_FILE = "chat_history.db"

def peek_recent_messages(limit=2):
    if not os.path.exists(DB_FILE):
        print(f"'{DB_FILE}' 파일이 존재하지 않습니다. 먼저 챗봇을 실행해 대화를 나눠보세요.")
        return

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        # id 기준 내림차순(DESC)으로 가장 최근 N개를 가져온 뒤, 시간순으로 보여주기 위해 역순 정렬
        cursor.execute(
            """
            SELECT id, session_id, role, content, created_at 
            FROM messages 
            ORDER BY id DESC 
            LIMIT ?
            """,
            (limit,)
        )
        rows = cursor.fetchall()

    if not rows:
        print("저장된 대화 내역이 없습니다.")
        return

    print(f"=== 가장 최근 대화 {len(rows)}줄 ===")
    for row in reversed(rows):
        msg_id, session_id, role, content, created_at = row
        speaker = "👤 사용자" if role == "user" else "🤖 AI"
        print(f"[{created_at}] ({speaker} / 방ID: {session_id})")
        print(f"내용: {content}")
        print("-" * 40)

if __name__ == "__main__":
    peek_recent_messages(limit=2)
