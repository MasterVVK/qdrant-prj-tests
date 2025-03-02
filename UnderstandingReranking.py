from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd

# Подключение к Qdrant
#client = QdrantClient(
#    url="<ADD-URL>",
#    api_key="<API-KEY>",
#)

client = QdrantClient(url="http://localhost:6333")
print(client.get_collections())

# Используем BGE-M3 для эмбеддингов и реранкинга
embedding_model = SentenceTransformer("BAAI/bge-large-en")  # BGE-M3 для эмбеддингов
rerank_model = CrossEncoder("BAAI/bge-reranker-large")  # BGE-M3 для реранкинга

embedding_model = SentenceTransformer("BAAI/bge-m3")  # BGE-M3 для эмбеддингов на русском
rerank_model = CrossEncoder("BAAI/bge-reranker-v2-m3")  # BGE-M3 для реранкинга

# Создание коллекции в Qdrant
if not client.collection_exists("basic-search-rerank"):
    client.create_collection(
	collection_name="basic-search-rerank",
	vectors_config=VectorParams(size=1024, distance=Distance.DOT),
    )

# Пример данных

query = "What is the purpose of feature scaling in machine learning?"

documents = [
    "In machine learning, feature scaling is the process of normalizing the range of independent variables or features. The goal is to ensure that all features contribute equally to the model, especially in algorithms like SVM or k-nearest neighbors where distance calculations matter.",
   
    "Feature scaling is commonly used in data preprocessing to ensure that features are on the same scale. This is particularly important for gradient descent-based algorithms where features with larger scales could disproportionately impact the cost function.",
   
    "In data science, feature extraction is the process of transforming raw data into a set of engineered features that can be used in predictive models. Feature scaling is related but focuses on adjusting the values of these features.",
   
    "Unsupervised learning algorithms, such as clustering methods, may benefit from feature scaling as it ensures that features with larger numerical ranges don't dominate the learning process.",
   
    "One common data preprocessing technique in data science is feature selection. Unlike feature scaling, feature selection aims to reduce the number of input variables used in a model to avoid overfitting.",
   
    "Principal component analysis (PCA) is a dimensionality reduction technique used in data science to reduce the number of variables. PCA works best when data is scaled, as it relies on variance which can be skewed by features on different scales.",
   
    "Min-max scaling is a common feature scaling technique that usually transforms features to a fixed range [0, 1]. This method is useful when the distribution of data is not Gaussian.",
   
    "Standardization, or z-score normalization, is another technique that transforms features into a mean of 0 and a standard deviation of 1. This method is effective for data that follows a normal distribution.",
   
    "Feature scaling is critical when using algorithms that rely on distances, such as k-means clustering, as unscaled features can lead to misleading results.",
   
    "Scaling can improve the convergence speed of gradient descent algorithms by preventing issues with different feature scales affecting the cost function's landscape.",
   
    "In deep learning, feature scaling helps in stabilizing the learning process, allowing for better performance and faster convergence during training.",
   
    "Robust scaling is another method that uses the median and the interquartile range to scale features, making it less sensitive to outliers.",
   
    "When working with time series data, feature scaling can help in standardizing the input data, improving model performance across different periods.",
   
    "Normalization is often used in image processing to scale pixel values to a range that enhances model performance in computer vision tasks.",
   
    "Feature scaling is significant when features have different units of measurement, such as height in centimeters and weight in kilograms.",
   
    "In recommendation systems, scaling features such as user ratings can improve the model's ability to find similar users or items.",
   
    "Dimensionality reduction techniques, like t-SNE and UMAP, often require feature scaling to visualize high-dimensional data in lower dimensions effectively.",
   
    "Outlier detection techniques can also benefit from feature scaling, as they can be influenced by unscaled features that have extreme values.",
   
    "Data preprocessing steps, including feature scaling, can significantly impact the performance of machine learning models, making it a crucial part of the modeling pipeline.",
   
    "In ensemble methods, like random forests, feature scaling is not strictly necessary, but it can still enhance interpretability and comparison of feature importance.",
   
    "Feature scaling should be applied consistently across training and test datasets to avoid data leakage and ensure reliable model evaluation.",
   
    "In natural language processing (NLP), scaling can be useful when working with numerical features derived from text data, such as word counts or term frequencies.",
   
    "Log transformation is a technique that can be applied to skewed data to stabilize variance and make the data more suitable for scaling.",
   
    "Data augmentation techniques in machine learning may also include scaling to ensure consistency across training datasets, especially in computer vision tasks."
]

# Генерация эмбеддингов с помощью BGE-M3
doc_embeddings = [embedding_model.encode(doc).tolist() for doc in documents]

# Запись эмбеддингов в Qdrant
points = [
    PointStruct(
        id=idx,
        vector=embedding,
        payload={"document": doc}
    )
    for idx, (embedding, doc) in enumerate(zip(doc_embeddings, documents))
]

operation_info = client.upsert(
    collection_name="basic-search-rerank",
    points=points
)

# Преобразование запроса в эмбеддинг
query_embedding = embedding_model.encode(query).tolist()

# Поиск векторных ближайших соседей
search_result = client.query_points(
    collection_name="basic-search-rerank", query=query_embedding, limit=10
).points

document_list = [point.payload['document'] for point in search_result]
document_ids = [point.id for point in search_result]
document_scores = [point.score for point in search_result]

# Вывод результатов поиска
search_results_df = pd.DataFrame({
    "ID": document_ids,
    "Document": [doc[:80] + "..." for doc in document_list],
    "Score": document_scores
})
print(search_results_df)

document_list = [point.payload['document'] for point in search_result]

# Реранкинг результатов с помощью BGE-M3
#scores = rerank_model.predict([(query, doc) for doc in document_list])
#reranked_results = [x for _, x in sorted(zip(scores, document_list), reverse=True)]

# Вывод результатов
#for i, doc in enumerate(reranked_results[:5]):
#    print(f"{i+1}. {doc}")
# Реранкинг результатов с помощью BGE-M3
scores = rerank_model.predict([(query, doc) for doc in document_list])
reranked_data = sorted(zip(document_ids, document_list, scores), key=lambda x: x[2], reverse=True)

# Вывод результатов реранкинга
reranked_results_df = pd.DataFrame({
    "ID": [doc_id for doc_id, _, _ in reranked_data],
    "Document": [doc[:80] + "..." for _, doc, _ in reranked_data],
    "Score": [score for _, _, score in reranked_data]
})
print(reranked_results_df)

