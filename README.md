# SpineLitAgent

척추 수술 문헌 리뷰 AI 에이전트 - 백엔드 시스템

## Features

- **Literature Discovery**: PubMed, Semantic Scholar, arXiv 통합 검색
- **Citation Analysis**: Importance score 기반 논문 평가
- **Dialectical Analysis**: Thesis-Antithesis-Synthesis 분석 (Claude API)
- **Knowledge Graph**: Neo4j 기반 논문 관계 그래프
- **Automation**: 일간/주간/월간 자동 업데이트

## Setup

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.template .env
# .env 파일에 API 키 입력

# Neo4j 실행 (Docker)
docker-compose up -d

# 서버 실행
uvicorn backend.main:app --reload
```

## API Endpoints

- `GET /api/papers/search` - 논문 검색
- `GET /api/papers/{id}/citations` - Citation 분석
- `POST /api/analysis/dialectic` - Dialectical synthesis
- `POST /api/graph/query` - Neo4j Cypher 쿼리
- `GET /api/health` - Health check

## License

MIT
