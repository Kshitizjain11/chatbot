from qdrant_client.models import VectorParams,Distance,PointStruct
from qdrant_client import QdrantClient

class QdrantStorage:
    def __init__(self,url='http://localhost:6333',collection="docs",dim=768):
        self.client = QdrantClient(url=url,timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(collection_name=self.collection,vectors_config=VectorParams(distance=Distance.COSINE,size=dim))
    
    def upsert(self,ids,vectors,payloads):
        points = [PointStruct(id=ids[i],vector=vectors[i],payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(collection_name=self.collection,points=points)

    def search(self,query_vector,top_k :int=5):
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )

        context = []
        sources = set()

        for r in results.points:
            payload = getattr(r,"payload",None) or {}
            text = payload.get("text","")
            source = payload.get("source","")

            if text:
                context.append(text)
                sources.add(source)
        
        return {"contexts":context,"sources":list(sources)}

