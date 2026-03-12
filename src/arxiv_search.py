# Arxiv API 검색 모듈

from langchain_community.document_loaders import ArxivLoader

loader = ArxivLoader(
    query="Chain of thought",
    load_max_docs=2,  # 최대 문서 수
    load_all_available_meta=True,  # 메타데이터 전체 로드 여부
)

docs = loader.load()
print(docs[0].metadata['Title'])

"""
metadata 키 값 설명
- 'Published': 논문이 출판된 날짜
- 'Title': 논문의 제목
- 'Authors': 논문의 저자 목록
- 'Summary': 논문의 Abstract
- 'entry_id': 논문의 고유 ID
- 'published_first_time': 논문 처음 출판 일자
- 'comment': 저자의 추가 코멘트
- 'journal_ref': 논문이 게재된 저널 정보
- 'doi': 논문의 DOI (Digital Object Identifier)
- 'primary_category': 논문의 주요 분야
- 'categories': 논문이 속한 분야 목록
- 'links': 논문의 아카이브 페이지, pdf 파일 링크 리스트

예시:
metadata={
'Published': '2024-10-04',
'Title': 'Understanding Reasoning in Chain-of-Thought from the Hopfieldian View',
'Authors': 'Lijie Hu, Liang Liu, Shu Yang, Xin Chen, Zhen Tan, Muhammad Asif Ali, Mengdi Li, Di Wang',
'Summary': "Large Language Models have demonstrated remarkable abilities across various tasks, with Chain-of-Thought (CoT) prompting emerging as a key technique to enhance reasoning capabilities. However, existing research primarily focuses on improving performance, lacking a comprehensive framework to explain and understand the fundamental factors behind CoT's success. (생략)", 
'entry_id': 'http://arxiv.org/abs/2410.03595v1',
'published_first_time': '2024-10-04',
'comment': '28 pages, a new version of "A Hopfieldian View-based Interpretation for Chain-of-Thought Reasoning"',
'journal_ref': None,
'doi': None,
'primary_category': 'cs.AI',
'categories': ['cs.AI', 'cs.CL', 'cs.LG'],
'links': ['https://arxiv.org/abs/2410.03595v1', 'https://arxiv.org/pdf/2410.03595v1']}
"""