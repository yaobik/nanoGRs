# ================================
# Codebook使用示例和扩展功能
# ================================

import numpy as np
import torch
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

# ================================
# 1. 快速入门版本
# ================================

def quick_start_codebook_training():
    """快速入门：最简单的codebook训练"""
    print("=== 快速入门版本 ===")
    
    # 创建简单的模拟数据
    np.random.seed(42)
    n_movies = 100
    n_features = 20
    
    # 模拟电影特征矩阵
    movie_features = np.random.randn(n_movies, n_features)
    movie_names = [f"Movie_{i}" for i in range(n_movies)]
    
    print(f"创建了 {n_movies} 部电影, 每部电影 {n_features} 个特征")
    
    # 简单的K-means聚类作为codebook
    from sklearn.cluster import KMeans
    
    # 训练codebook
    n_clusters = 16  # codebook大小
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(movie_features)
    
    # 每个电影的code就是它的cluster标签
    print(f"训练完成! 创建了 {n_clusters} 个codes")
    print(f"电影codes分布: {Counter(cluster_labels)}")
    
    # 查看一些例子
    for i in range(5):
        print(f"{movie_names[i]}: code = {cluster_labels[i]}")
    
    return kmeans, cluster_labels, movie_features, movie_names

# ================================
# 2. Codebook应用示例
# ================================

class CodebookRecommender:
    """基于Codebook的推荐器"""
    
    def __init__(self, model, codes, movie_data, features):
        self.model = model
        self.codes = codes
        self.movie_data = movie_data
        self.features = features
        
    def find_similar_movies_by_code(self, movie_id: int, top_k: int = 5):
        """基于相同code找相似电影"""
        movie_code = self.codes[movie_id]
        
        # 找到有相同code的其他电影
        similar_mask = np.all(self.codes == movie_code, axis=1)
        similar_indices = np.where(similar_mask)[0]
        
        # 排除自己
        similar_indices = similar_indices[similar_indices != movie_id]
        
        if len(similar_indices) == 0:
            return []
        
        # 计算特征相似度进行排序
        target_features = self.features[movie_id].reshape(1, -1)
        similar_features = self.features[similar_indices]
        
        similarities = cosine_similarity(target_features, similar_features)[0]
        
        # 排序并返回top_k
        sorted_indices = np.argsort(similarities)[::-1][:top_k]
        result_indices = similar_indices[sorted_indices]
        
        results = []
        for idx in result_indices:
            results.append({
                'movie_id': idx,
                'title': self.movie_data.iloc[idx]['title'],
                'similarity': similarities[sorted_indices[len(results)]],
                'code': self.codes[idx]
            })
        
        return results
    
    def analyze_code_clusters(self):
        """分析每个code代表的电影类型"""
        analysis = {}
        
        for layer in range(self.codes.shape[1]):
            layer_codes = self.codes[:, layer]
            unique_codes = np.unique(layer_codes)
            
            layer_analysis = {}
            for code in unique_codes:
                mask = layer_codes == code
                movies_in_cluster = self.movie_data[mask]
                
                # 分析类型分布
                all_genres = []
                for genres_str in movies_in_cluster['genres']:
                    all_genres.extend(genres_str.split('|'))
                
                genre_counts = Counter(all_genres)
                
                # 分析年代分布
                years = movies_in_cluster['year'].values
                
                layer_analysis[code] = {
                    'count': mask.sum(),
                    'top_genres': genre_counts.most_common(3),
                    'avg_year': np.mean(years),
                    'avg_rating': movies_in_cluster['vote_average'].mean()
                }
            
            analysis[f'layer_{layer}'] = layer_analysis
        
        return analysis
    
    def compress_user_history(self, user_movie_ids: list):
        """压缩用户历史为code序列"""
        user_codes = []
        for movie_id in user_movie_ids:
            if movie_id < len(self.codes):
                user_codes.append(self.codes[movie_id])
        
        return np.array(user_codes)
    
    def generate_recommendations(self, user_movie_ids: list, top_k: int = 10):
        """基于用户历史生成推荐"""
        # 获取用户看过的电影的codes
        user_codes = self.compress_user_history(user_movie_ids)
        
        if len(user_codes) == 0:
            return []
        
        # 计算每个code的权重（基于用户偏好）
        code_weights = {}
        for layer in range(user_codes.shape[1]):
            layer_codes = user_codes[:, layer]
            layer_counter = Counter(layer_codes)
            
            for code, count in layer_counter.items():
                if code not in code_weights:
                    code_weights[code] = 0
                code_weights[code] += count
        
        # 为所有电影计算推荐分数
        scores = np.zeros(len(self.movie_data))
        
        for movie_id in range(len(self.movie_data)):
            if movie_id in user_movie_ids:
                continue  # 跳过已看过的电影
            
            movie_code = self.codes[movie_id]
            score = 0
            
            for layer in range(len(movie_code)):
                code = movie_code[layer]
                if code in code_weights:
                    score += code_weights[code]
            
            scores[movie_id] = score
        
        # 获取top_k推荐
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有正分数的
                recommendations.append({
                    'movie_id': idx,
                    'title': self.movie_data.iloc[idx]['title'],
                    'score': scores[idx],
                    'code': self.codes[idx]
                })
        
        return recommendations

# ================================
# 3. 可视化工具
# ================================

def visualize_codebook_analysis(codes, movie_data):
    """可视化codebook分析"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Code分布直方图
    for layer in range(min(2, codes.shape[1])):
        layer_codes = codes[:, layer]
        axes[0, layer].hist(layer_codes, bins=50, alpha=0.7)
        axes[0, layer].set_title(f'Layer {layer+1} Code Distribution')
        axes[0, layer].set_xlabel('Code Value')
        axes[0, layer].set_ylabel('Frequency')
    
    # 2. 电影年代vs Code的关系
    if codes.shape[1] >= 1:
        scatter_data = pd.DataFrame({
            'year': movie_data['year'],
            'code': codes[:, 0],
            'rating': movie_data['vote_average']
        })
        
        axes[1, 0].scatter(scatter_data['year'], scatter_data['code'], 
                          c=scatter_data['rating'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('Movie Year vs Code (colored by rating)')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Code')
        
        # 添加colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array(scatter_data['rating'])
        plt.colorbar(sm, ax=axes[1, 0])
    
    # 3. Code利用率
    if codes.shape[1] >= 1:
        layer_codes = codes[:, 0]
        unique_codes, counts = np.unique(layer_codes, return_counts=True)
        
        axes[1, 1].bar(range(len(unique_codes)), counts)
        axes[1, 1].set_title('Code Usage Distribution')
        axes[1, 1].set_xlabel('Code Index')
        axes[1, 1].set_ylabel('Usage Count')
    
    plt.tight_layout()
    plt.show()

def plot_movie_similarity_matrix(recommender: CodebookRecommender, sample_movies: list):
    """绘制电影相似度矩阵"""
    n_movies = len(sample_movies)
    similarity_matrix = np.zeros((n_movies, n_movies))
    
    movie_titles = [recommender.movie_data.iloc[i]['title'] for i in sample_movies]
    
    for i, movie_i in enumerate(sample_movies):
        for j, movie_j in enumerate(sample_movies):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # 计算code相似度
                code_i = recommender.codes[movie_i]
                code_j = recommender.codes[movie_j]
                
                # 简单的相似度：相同位置的匹配数
                similarity = np.mean(code_i == code_j)
                similarity_matrix[i, j] = similarity
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                xticklabels=[t[:20] for t in movie_titles],
                yticklabels=[t[:20] for t in movie_titles],
                annot=True, cmap='Blues', fmt='.2f')
    plt.title('Movie Similarity Matrix (based on Codes)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ================================
# 4. 使用示例
# ================================

def demo_codebook_applications():
    """演示codebook的各种应用"""
    print("=== Codebook应用演示 ===\n")
    
    # 假设我们已经有了训练好的模型和数据
    # 这里使用快速入门版本作为演示
    kmeans_model, cluster_labels, features, movie_names = quick_start_codebook_training()
    
    # 创建模拟的movie_data
    movie_data = pd.DataFrame({
        'title': movie_names,
        'genres': ['Action|Drama'] * len(movie_names),
        'year': np.random.randint(1990, 2024, len(movie_names)),
        'vote_average': np.random.uniform(5, 9, len(movie_names))
    })
    
    # 将cluster_labels转换为codes格式
    codes = cluster_labels.reshape(-1, 1)  # 单层codebook
    
    # 创建推荐器
    recommender = CodebookRecommender(kmeans_model, codes, movie_data, features)
    
    print("\n1. 相似电影推荐:")
    similar_movies = recommender.find_similar_movies_by_code(0, top_k=3)
    for movie in similar_movies:
        print(f"  - {movie['title']} (相似度: {movie['similarity']:.3f})")
    
    print("\n2. 基于用户历史的推荐:")
    user_history = [0, 1, 2]  # 用户看过的电影ID
    recommendations = recommender.generate_recommendations(user_history, top_k=5)
    for rec in recommendations:
        print(f"  - {rec['title']} (分数: {rec['score']})")
    
    print("\n3. 用户历史压缩:")
    compressed_history = recommender.compress_user_history(user_history)
    print(f"  原始历史: {user_history}")
    print(f"  压缩后: {compressed_history.flatten()}")
    
    return recommender

# ================================
# 5. 实际数据集加载器
# ================================

def load_real_movielens_data(data_path: str):
    """加载真实的MovieLens数据集"""
    try:
        # 读取数据文件
        movies = pd.read_csv(f'{data_path}/movies.csv')
        ratings = pd.read_csv(f'{data_path}/ratings.csv')
        
        print(f"成功加载MovieLens数据:")
        print(f"  - 电影数量: {len(movies)}")
        print(f"  - 评分数量: {len(ratings)}")
        print(f"  - 用户数量: {ratings['userId'].nunique()}")
        
        return movies, ratings
        
    except FileNotFoundError:
        print(f"未找到数据文件在路径: {data_path}")
        print("请下载MovieLens数据集或使用模拟数据")
        return None, None

# ================================
# 运行演示
# ================================

if __name__ == "__main__":
    # 演示各种功能
    recommender = demo_codebook_applications()
    
    print("\n=== 可用功能总结 ===")
    print("1. 训练电影特征的codebook")
    print("2. 基于code的相似电影推荐")
    print("3. 用户历史压缩和推荐生成")
    print("4. Codebook cluster分析")
    print("5. 可视化工具")
    print("\n使用方法:")
    print("recommender.find_similar_movies_by_code(movie_id)")
    print("recommender.generate_recommendations(user_history)")
    print("recommender.analyze_code_clusters()")