from fastembed import TextEmbedding, LateInteractionTextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
import pandas as pd

print(LateInteractionTextEmbedding.list_supported_models())

client = QdrantClient(url="http://localhost:6333")

dense_embedding_model = TextEmbedding("intfloat/multilingual-e5-large")
late_interaction_embedding_model = LateInteractionTextEmbedding("jinaai/jina-colbert-v2")
bm25_embedding_model = SparseTextEmbedding("Qdrant/bm25")



query = "Какова цель масштабирования признаков в машинном обучении?"

documents = [
    "В машинном обучении масштабирование признаков – это процесс нормализации диапазона независимых переменных или признаков. Его цель – обеспечить равномерный вклад всех признаков в модель, особенно в алгоритмах, таких как SVM или k-ближайших соседей, где важны вычисления расстояний.",

    "Масштабирование признаков часто используется на этапе предварительной обработки данных, чтобы привести признаки к одному масштабу. Это особенно важно для алгоритмов, основанных на градиентном спуске, так как признаки с большими значениями могут непропорционально влиять на функцию стоимости.",

    "В науке о данных процесс извлечения признаков включает преобразование необработанных данных в набор инженерных признаков, используемых в моделях прогнозирования. Масштабирование признаков связано с этим процессом, но его основная цель – корректировка значений этих признаков.",

    "Алгоритмы обучения без учителя, такие как методы кластеризации, могут выигрывать от масштабирования признаков, поскольку оно предотвращает доминирование признаков с большими числовыми диапазонами.",

    "Одна из популярных техник предварительной обработки данных – отбор признаков. В отличие от масштабирования, он направлен на уменьшение количества входных переменных модели для предотвращения переобучения.",

    "Анализ главных компонент (PCA) – это метод уменьшения размерности, который работает лучше при масштабированных данных, поскольку он основан на дисперсии, которая может быть искажена признаками с разными масштабами.",

    "Мин-макс нормализация – это популярный метод масштабирования, который обычно приводит значения признаков к фиксированному диапазону [0, 1]. Этот метод особенно полезен, когда данные не имеют нормального распределения.",

    "Стандартизация (z-score нормализация) – еще один метод, который преобразует признаки таким образом, чтобы их среднее значение было 0, а стандартное отклонение – 1. Этот метод эффективен, если данные имеют нормальное распределение.",

    "Масштабирование признаков критически важно при использовании алгоритмов, основанных на расстояниях, например, в k-средних кластеризации, поскольку немасштабированные признаки могут приводить к некорректным результатам.",

    "Градиентный спуск работает эффективнее при масштабировании, так как различия в масштабе признаков могут замедлить сходимость алгоритма.",

    "В глубоком обучении масштабирование признаков помогает стабилизировать процесс обучения, обеспечивая быструю и стабильную сходимость моделей.",

    "Робастное масштабирование – еще один метод, использующий медиану и межквартильный размах для масштабирования признаков, что делает его менее чувствительным к выбросам.",

    "При работе с временными рядами масштабирование помогает стандартизировать входные данные и улучшить производительность модели на разных временных периодах.",

    "Нормализация часто используется в обработке изображений для приведения значений пикселей в определенный диапазон, что улучшает работу моделей компьютерного зрения.",

    "Масштабирование особенно важно, когда признаки имеют разные единицы измерения, например, рост в сантиметрах и вес в килограммах.",

    "В системах рекомендаций масштабирование признаков, таких как рейтинги пользователей, может улучшить способность модели находить похожих пользователей или объекты.",

    "Методы уменьшения размерности, такие как t-SNE и UMAP, требуют масштабирования признаков для корректной визуализации многомерных данных в меньших измерениях.",

    "Методы обнаружения выбросов могут выигрывать от масштабирования, так как выбросы могут быть усилены или ослаблены признаками с разными масштабами.",

    "Предварительная обработка данных, включая масштабирование, может существенно повлиять на производительность моделей машинного обучения, поэтому это важный этап подготовки данных.",

    "В ансамблевых методах, таких как случайный лес, масштабирование не является строго обязательным, но может улучшить интерпретируемость модели и сравнение значимости признаков.",

    "Масштабирование должно применяться одинаково к обучающим и тестовым данным, чтобы избежать утечки данных и обеспечить корректную оценку модели.",

    "В обработке естественного языка (NLP) масштабирование может быть полезно при работе с числовыми признаками из текстов, например, частотой встречаемости слов.",

    "Логарифмическое преобразование можно применять к асимметричным данным для стабилизации дисперсии и улучшения процесса масштабирования.",

    "В методах увеличения данных (data augmentation) масштабирование может использоваться для обеспечения консистентности обучающего набора, особенно в задачах компьютерного зрения."
]

dense_embeddings = list(dense_embedding_model.embed(doc for doc in documents))
bm25_embeddings = list(bm25_embedding_model.embed(doc for doc in documents))
late_interaction_embeddings = list(late_interaction_embedding_model.embed(doc for doc in documents))

from qdrant_client.models import Distance, VectorParams, models

if not client.collection_exists("hybrid-search_RU"):
    client.create_collection(
        "hybrid-search_RU",
        vectors_config={
            "all-MiniLM-L6-v2": models.VectorParams(
                size=len(dense_embeddings[0]),
                distance=models.Distance.COSINE,
            ),
            "colbertv2.0": models.VectorParams(
                size=len(late_interaction_embeddings[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                )
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF
            )
        }
    )

from qdrant_client.models import PointStruct

points = []
for idx, (dense_embedding, bm25_embedding, late_interaction_embedding, doc) in enumerate(
        zip(dense_embeddings, bm25_embeddings, late_interaction_embeddings, documents)):
    point = PointStruct(
        id=idx,
        vector={
            "all-MiniLM-L6-v2": dense_embedding,
            "bm25": bm25_embedding.as_object(),
            "colbertv2.0": late_interaction_embedding,
        },
        payload={"document": doc}
    )
    points.append(point)

operation_info = client.upsert(
    collection_name="hybrid-search_RU",
    points=points
)

dense_vectors = next(dense_embedding_model.query_embed(query))
sparse_vectors = next(bm25_embedding_model.query_embed(query))
late_vectors = next(late_interaction_embedding_model.query_embed(query))

prefetch = [
        models.Prefetch(
            query=dense_vectors,
            using="all-MiniLM-L6-v2",
            limit=20,
        ),
        models.Prefetch(
            query=models.SparseVector(**sparse_vectors.as_object()),
            using="bm25",
            limit=20,
        ),
    ]
results = client.query_points(
         "hybrid-search_RU",
        prefetch=prefetch,
        query=late_vectors,
        using="colbertv2.0",
        with_payload=True,
        limit=10,
)


#print(results)

# Форматированный вывод результатов
search_results = [
    {
        "ID": point.id,
        "Document": point.payload["document"],
        "Score": round(point.score, 2)
    }
    for point in results.points
]

search_results_df = pd.DataFrame(search_results)
print(search_results_df)