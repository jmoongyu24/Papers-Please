# Arxiv API 검색 모듈
from langchain_community.retrievers import ArxivRetriever

retriever = ArxivRetriever(
    load_max_docs=1,
    load_all_available_meta=True,
    get_full_documents=False,
)

docs = retriever.invoke("What is Transformer model?")

for doc in docs:
    print(doc.metadata.get("Title"))
    print(doc)
    print()

"""
page_content: Abstract
metadata 키 값 설명
- 'Entry ID': ArXiv 주소랑 ID / 'http://arxiv.org/abs/2202.09741v5' 이런 식
- 'Published': 출판 날짜 / datetime.date(2022, 7, 11) 이런 식
- 'Title': 논문 제목 / 'Visual Attention Network' 이런 식
- 'Authors': 논문 작성자 / 'Meng-Hao Guo, Cheng-Ze Lu, Zheng-Ning Liu, Ming-Ming Cheng, Shi-Min Hu' 이런 식
"""