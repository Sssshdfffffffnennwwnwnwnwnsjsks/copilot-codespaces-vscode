ecommerce-ai/
├── frontend/           # واجهة المستخدم (React)
├── backend/            # الخادم (Node.js/Express)
├── admin-panel/        # لوحة التحكم (React)
├── ai-modules/         # نماذج الذكاء الاصطناعي (Python)
└── database/           # قاعدة البيانات (MongoDB)
# ai-modules/recommendation_engine.py
import numpy as np
from sklearn.neighbors import NearestNeighbors

class RecommendationEngine:
    def __init__(self):
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.product_embeddings = {}
        self.product_ids = []
        
    def train(self, products):
        """تدريب النموذج على بيانات المنتجات"""
        embeddings = []
        for product in products:
            embedding = self._generate_embedding(product)
            self.product_embeddings[product['id']] = embedding
            self.product_ids.append(product['id'])
            embeddings.append(embedding)
            
        self.model.fit(np.array(embeddings))
    
    def _generate_embedding(self
