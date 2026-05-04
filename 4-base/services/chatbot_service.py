"""
🎯 챗봇 서비스 - 피코(PICO) 구현

프로메테우스 호의 메인 AI '피코'의 핵심 로직을 담당합니다.
RAG 기반 검색과 OpenAI LLM을 활용하여 방탈출 추리 게임을 진행합니다.

📐 시스템 아키텍처:

┌─────────────────────────────────────────────────────────┐
│ 1. 초기화 단계 (ChatbotService.__init__)                  │
├─────────────────────────────────────────────────────────┤
│  - OpenAI Client 생성                                    │
│  - ChromaDB 연결 (벡터 데이터베이스)                       │
│  - LangChain Memory 초기화 (대화 기록 관리)               │
│  - Config 파일 로드                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 2. RAG 파이프라인 (generate_response 내부)               │
├─────────────────────────────────────────────────────────┤
│  사용자 질문 → _create_embedding() → 벡터 변환           │
│       → _search_similar() → ChromaDB 유사 문서 검색      │
│       → _build_prompt() → 시스템 + RAG + 질문 결합       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 3. LLM 응답 생성 → 메모리 저장 → 응답 반환               │
└─────────────────────────────────────────────────────────┘
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent


class ChatbotService:
    """
    챗봇 서비스 클래스

    이 클래스는 챗봇의 모든 AI 로직을 캡슐화합니다.

    주요 책임:
    1. OpenAI API 관리
    2. ChromaDB 벡터 검색 (RAG)
    3. LangChain 메모리 관리
    4. 응답 생성 파이프라인
    """

    def __init__(self):
        """
        챗봇 서비스 초기화

        1. Config 로드 (chatbot_config.json)
        2. OpenAI Client 생성
        3. ChromaDB 연결
        4. LangChain Memory 초기화
        """
        print("[ChatbotService] 초기화 중...")

        # 1. Config 로드
        self.config = self._load_config()
        print(f"  [Config] 챗봇 이름: {self.config.get('name', '알 수 없음')}")

        # 2. OpenAI Client 생성
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        self.client = OpenAI(api_key=api_key)
        print("  [OpenAI] 클라이언트 생성 완료")

        # 3. ChromaDB 연결
        self.collection = self._init_chromadb()

        # 4. LangChain Memory 초기화 (대화 기록 관리)
        self.memory = self._init_memory()

        print("[ChatbotService] 초기화 완료")

    def _load_config(self):
        """
        설정 파일 로드

        config/chatbot_config.json 파일을 읽어서 dict로 반환합니다.
        """
        config_path = BASE_DIR / "config" / "chatbot_config.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print("  [WARNING] chatbot_config.json 을 찾을 수 없습니다. 기본 설정을 사용합니다.")
            return {
                "name": "챗봇",
                "description": "",
                "system_prompt": {"base": "", "rules": []},
            }

    def _init_chromadb(self):
        """
        ChromaDB 초기화 및 컬렉션 반환

        1. PersistentClient 생성
        2. 컬렉션 가져오기 (이름: "rag_collection")
        3. 컬렉션 반환
        """
        db_path = BASE_DIR / "static" / "data" / "chatbot" / "chardb_embedding"
        try:
            client = chromadb.PersistentClient(path=str(db_path))
            collection = client.get_collection(name="rag_collection")
            print(f"  [ChromaDB] 컬렉션 로드 완료 — 문서 수: {collection.count()}")
            return collection
        except Exception as e:
            print(f"  [WARNING] ChromaDB 초기화 실패: {e}")
            print("  [WARNING] RAG 검색 없이 일반 대화 모드로 작동합니다.")
            return None

    def _init_memory(self):
        """
        LangChain 기반 대화 메모리 초기화

        ConversationSummaryBufferMemory 를 사용하여
        대화가 길어지면 오래된 대화를 자동으로 요약합니다.
        사용 불가 시 간단한 리스트 기반 버퍼로 대체합니다.
        """
        try:
            from langchain_openai import ChatOpenAI
            from langchain.memory import ConversationSummaryBufferMemory

            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
            memory = ConversationSummaryBufferMemory(
                llm=llm,
                max_token_limit=300,
                return_messages=False,
                memory_key="history",
            )
            print("  [Memory] LangChain ConversationSummaryBufferMemory 초기화 완료")
            return memory
        except ImportError:
            print("  [Memory] LangChain Memory 사용 불가 — 리스트 버퍼로 대체합니다.")
            return None

    # ================================================================
    # 대화 버퍼 (LangChain Memory 사용 불가 시 폴백)
    # ================================================================
    _conversation_buffer: list = []
    _BUFFER_MAX_SIZE = 10  # 최근 10개 메시지(5회 대화)

    def _save_to_buffer(self, user_message: str, bot_reply: str):
        """대화 기록을 메모리(또는 폴백 버퍼)에 저장"""
        if self.memory is not None:
            # LangChain Memory 사용
            self.memory.save_context(
                {"input": user_message},
                {"output": bot_reply},
            )
        else:
            # 폴백: 리스트 버퍼
            self._conversation_buffer.append({"role": "user", "content": user_message})
            self._conversation_buffer.append({"role": "assistant", "content": bot_reply})
            if len(self._conversation_buffer) > self._BUFFER_MAX_SIZE:
                self._conversation_buffer = self._conversation_buffer[-self._BUFFER_MAX_SIZE:]

    def _get_history_text(self) -> str:
        """대화 기록을 텍스트로 반환"""
        if self.memory is not None:
            try:
                variables = self.memory.load_memory_variables({})
                return variables.get("history", "")
            except Exception:
                return ""
        else:
            if not self._conversation_buffer:
                return ""
            lines = []
            for msg in self._conversation_buffer[-6:]:  # 최근 3회 대화
                role = "사용자" if msg["role"] == "user" else "피코"
                lines.append(f"{role}: {msg['content']}")
            return "\n".join(lines)

    # ================================================================
    # Embedding & RAG
    # ================================================================

    def _create_embedding(self, text: str) -> list:
        """
        텍스트를 임베딩 벡터로 변환

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            list: 3072차원 벡터 (text-embedding-3-large 모델)
        """
        response = self.client.embeddings.create(
            input=[text],
            model="text-embedding-3-large",
        )
        return response.data[0].embedding

    def _search_similar(self, query: str, threshold: float = 0.40, top_k: int = 5):
        """
        RAG 검색: 유사한 문서 찾기 (핵심 메서드!)

        Args:
            query (str): 검색 질의
            threshold (float): 유사도 임계값
            top_k (int): 검색할 문서 개수

        Returns:
            tuple: (document, similarity, metadata) 또는 (None, None, None)

        유사도 공식: similarity = 1 / (1 + distance)
        - distance 가 작을수록 유사 → similarity 가 클수록 유사
        """
        if not self.collection:
            return None, None, None

        # 1. 쿼리 임베딩 생성
        query_embedding = self._create_embedding(query)

        # 2. ChromaDB 검색
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        # 3. 유사도 계산 및 필터링
        if not results["documents"] or not results["documents"][0]:
            return None, None, None

        best_doc = None
        best_similarity = 0.0
        best_meta = None

        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            similarity = 1 / (1 + dist)

            print(f"    [RAG] [{meta.get('title', '?')}] "
                  f"거리={dist:.4f}  유사도={similarity:.4f}")

            if similarity >= threshold and similarity > best_similarity:
                best_doc = doc
                best_similarity = similarity
                best_meta = meta

        # 4. 가장 유사한 문서 반환
        if best_doc:
            return best_doc, best_similarity, best_meta
        return None, None, None

    # ================================================================
    # Prompt Building
    # ================================================================

    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자"):
        """
        LLM 프롬프트 구성

        시스템 프롬프트는 별도로 system role 로 전달하므로,
        여기서는 RAG 컨텍스트 + 대화 기록 + 사용자 메시지를 결합합니다.

        Args:
            user_message (str): 사용자 메시지
            context (str): RAG 검색 결과 (선택)
            username (str): 사용자 이름

        Returns:
            str: 최종 사용자 프롬프트
        """
        parts = []

        # RAG 컨텍스트 (있을 때만)
        if context:
            parts.append(f"[참고 정보]\n{context}")

        # 대화 기록
        history = self._get_history_text()
        if history:
            parts.append(f"[이전 대화]\n{history}")

        # 사용자 메시지
        parts.append(f"사용자({username}): {user_message}")

        return "\n\n".join(parts)

    def _build_system_message(self) -> str:
        """
        시스템 프롬프트 구성

        config 의 system_prompt.base + rules 를 결합합니다.
        """
        system_prompt = self.config.get("system_prompt", {})
        base = system_prompt.get("base", "")
        rules = system_prompt.get("rules", [])

        parts = [base]
        if rules:
            rules_text = "\n".join(f"- {rule}" for rule in rules)
            parts.append(f"\n규칙:\n{rules_text}")

        return "\n".join(parts)

    # ================================================================
    # 응답 생성 파이프라인
    # ================================================================

    def generate_response(self, user_message: str, username: str = "사용자") -> dict:
        """
        사용자 메시지에 대한 챗봇 응답 생성

        Args:
            user_message (str): 사용자 입력
            username (str): 사용자 이름

        Returns:
            dict: {
                'reply': str,       # 챗봇 응답 텍스트
                'image': str|None   # 이미지 경로 (선택)
            }
        """
        try:
            # ──────────────────────────────────────────────
            # [1단계] 초기 메시지 처리
            # ──────────────────────────────────────────────
            if user_message.strip().lower() == "init":
                bot_name = self.config.get("name", "챗봇")
                init_reply = (
                    f"...삐빅. 시스템 부팅 중...\n\n"
                    f"어라? 깨어나셨군요! 저는 {bot_name}, "
                    f"프로메테우스 호의 메인 AI예요.\n\n"
                    f"소행성 충돌 때문에 우주선이 좀 엉망이 됐는데... "
                    f"그보다 당신, 혹시 자기가 누군지 기억나세요?\n\n"
                    f"아닌 것 같은 표정이네요. 괜찮아요, 제가 도와드릴게요! "
                    f"시스템 복구를 위해 당신의 직업과 이름을 확인해야 해요.\n\n"
                    f"먼저 직업부터 알아내 볼까요? "
                    f"우주선 이곳저곳을 둘러보면 단서를 찾을 수 있을 거예요!"
                )
                return {"reply": init_reply, "image": None}

            # ──────────────────────────────────────────────
            # [2단계] RAG 검색 수행
            # ──────────────────────────────────────────────
            print(f"\n{'='*50}")
            print(f"[USER] {username}: {user_message}")
            print(f"[RAG] 검색 중...")

            context, similarity, metadata = self._search_similar(
                query=user_message,
                threshold=0.40,
                top_k=5,
            )
            has_context = context is not None

            if has_context:
                print(f"[RAG] ✅ 매칭됨 — [{metadata.get('title', '?')}] 유사도: {similarity:.4f}")
                print(f"[RAG] 컨텍스트: {context[:100]}...")
            else:
                print("[RAG] ❌ threshold 이상 매칭 없음 — 일반 대화 모드")

            # ──────────────────────────────────────────────
            # [3단계] 프롬프트 구성
            # ──────────────────────────────────────────────
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username,
            )
            system_message = self._build_system_message()

            # ──────────────────────────────────────────────
            # [4단계] LLM API 호출
            # ──────────────────────────────────────────────
            print("[LLM] API 호출 중...")

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8,
                max_tokens=500,
            )

            reply = response.choices[0].message.content

            # ──────────────────────────────────────────────
            # [5단계] 메모리 저장
            # ──────────────────────────────────────────────
            self._save_to_buffer(user_message, reply)

            # ──────────────────────────────────────────────
            # [6단계] 응답 반환
            # ──────────────────────────────────────────────
            print(f"[BOT] {reply}")
            print(f"{'='*50}\n")

            return {
                "reply": reply,
                "image": None,  # 이미지 검색 로직 추가 가능
            }

        except Exception as e:
            print(f"[ERROR] 응답 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return {
                "reply": "삐빅... 시스템에 일시적인 오류가 발생했어요. 다시 한 번 말씀해주시겠어요?",
                "image": None,
            }


# ============================================================================
# 싱글톤 패턴
# ============================================================================
# ChatbotService 인스턴스를 앱 전체에서 재사용
# (매번 새로 초기화하면 비효율적)

_chatbot_service = None


def get_chatbot_service():
    """
    챗봇 서비스 인스턴스 반환 (싱글톤)

    첫 호출 시 인스턴스 생성, 이후 재사용
    """
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service


# ============================================================================
# 테스트용 메인 함수
# ============================================================================

if __name__ == "__main__":
    """
    로컬 테스트용

    실행 방법:
    python services/chatbot_service.py
    """
    print("챗봇 서비스 테스트")
    print("=" * 50)

    service = get_chatbot_service()

    # 초기화 테스트
    response = service.generate_response("init", "테스터")
    print(f"초기 응답: {response}")

    # 일반 대화 테스트
    response = service.generate_response("안녕하세요!", "테스터")
    print(f"응답: {response}")

    # RAG 테스트
    response = service.generate_response("내 방에 뭐가 있어?", "테스터")
    print(f"응답: {response}")

    # 직업 질문 테스트
    response = service.generate_response("내 직업이 뭐야?", "테스터")
    print(f"응답: {response}")
