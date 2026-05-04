"""
임베딩 빌드 스크립트

텍스트 데이터를 읽어서 OpenAI 임베딩으로 변환한 뒤 ChromaDB에 저장합니다.

사용법:
    python build_embedding.py

텍스트 수정 후 다시 실행하면 기존 DB를 삭제하고 새로 만듭니다.
"""

import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# 환경변수 로드
load_dotenv()

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "static" / "data" / "chatbot" / "chardb_text"
EMBEDDING_DIR = BASE_DIR / "static" / "data" / "chatbot" / "chardb_embedding"
COLLECTION_NAME = "rag_collection"


def load_text_files():
    """
    chardb_text/ 폴더의 모든 텍스트 파일을 읽어서
    [제목] 태그 기준으로 청크 단위로 나눕니다.
    """
    chunks = []

    for filepath in sorted(TEXT_DIR.glob("*.txt")):
        print(f"  📄 읽는 중: {filepath.name}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # [제목] 태그 기준으로 섹션 분리
        sections = content.split("[")
        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 제목과 본문 분리
            if "]" in section:
                title, body = section.split("]", 1)
                title = title.strip()
                body = body.strip()
            else:
                title = "기타"
                body = section.strip()

            if not body:
                continue

            chunks.append({
                "title": title,
                "text": body,
                "source": filepath.name,
            })

    return chunks


def create_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    OpenAI API를 사용하여 텍스트 목록을 임베딩 벡터로 변환합니다.
    한 번에 최대 20개씩 배치로 처리합니다.
    """
    all_embeddings = []
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  🔄 임베딩 생성 중... ({i + 1}~{i + len(batch)}/{len(texts)})")

        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large"
        )

        for item in response.data:
            all_embeddings.append(item.embedding)

    return all_embeddings


def build_chromadb(chunks, embeddings):
    """
    ChromaDB에 임베딩 데이터를 저장합니다.
    기존 DB가 있으면 삭제하고 새로 만듭니다.
    """
    # 기존 DB 삭제
    if EMBEDDING_DIR.exists() and any(EMBEDDING_DIR.iterdir()):
        print("  🗑️  기존 ChromaDB 삭제 중...")
        shutil.rmtree(EMBEDDING_DIR)
        EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    # ChromaDB 클라이언트 생성
    chroma_client = chromadb.PersistentClient(path=str(EMBEDDING_DIR))

    # 컬렉션 생성
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "l2"}  # L2 거리 (유클리드)
    )

    # 데이터 추가
    ids = [str(i) for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "title": chunk["title"],
            "source": chunk["source"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def main():
    print("=" * 50)
    print("🚀 임베딩 빌드 시작")
    print("=" * 50)

    # 1. API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return

    openai_client = OpenAI(api_key=api_key)

    # 2. 텍스트 파일 로드
    print("\n📂 텍스트 파일 로드 중...")
    chunks = load_text_files()
    print(f"  ✅ 총 {len(chunks)}개 청크 로드 완료")

    if not chunks:
        print("❌ chardb_text/ 폴더에 텍스트 파일이 없습니다.")
        return

    # 청크 목록 출력
    print("\n📋 청크 목록:")
    for i, chunk in enumerate(chunks):
        preview = chunk["text"][:50].replace("\n", " ")
        print(f"  [{i}] [{chunk['title']}] {preview}...")

    # 3. 임베딩 생성
    print("\n🧠 임베딩 생성 중...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(openai_client, texts)
    print(f"  ✅ {len(embeddings)}개 임베딩 생성 완료 (차원: {len(embeddings[0])})")

    # 4. ChromaDB 저장
    print("\n💾 ChromaDB에 저장 중...")
    collection = build_chromadb(chunks, embeddings)
    print(f"  ✅ 컬렉션 '{COLLECTION_NAME}'에 {collection.count()}개 문서 저장 완료")

    # 5. 검증 — 테스트 쿼리
    print("\n🔍 테스트 검색 수행 중...")
    test_queries = ["내 직업이 뭐야?", "내 방에 뭐가 있어?", "여기가 어디야?"]

    for query in test_queries:
        query_embedding = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-large"
        ).data[0].embedding

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "distances", "metadatas"]
        )

        print(f"\n  Q: \"{query}\"")
        for j in range(len(results["documents"][0])):
            doc = results["documents"][0][j][:60].replace("\n", " ")
            dist = results["distances"][0][j]
            sim = 1 / (1 + dist)
            title = results["metadatas"][0][j]["title"]
            print(f"    → [{title}] 유사도: {sim:.4f} | {doc}...")

    print("\n" + "=" * 50)
    print("✅ 빌드 완료!")
    print(f"   DB 경로: {EMBEDDING_DIR}")
    print(f"   다음에 텍스트를 수정하면 이 스크립트를 다시 실행하세요.")
    print("=" * 50)


if __name__ == "__main__":
    main()
