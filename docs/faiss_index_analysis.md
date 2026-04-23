# FAISSIndex 向量索引实现分析

## 核心结构

`FAISSIndex` 类封装了 FAISS 库，提供四种索引类型：

| 索引类型 | 说明 | 特点 |
|----------|------|------|
| `Flat` | `IndexFlatIP` | 精确暴力搜索，适合小数据集 |
| `IVF` | `IndexIVFFlat` | 倒排文件索引，先聚类再搜索，加速但可能有精度损失 |
| `IVFPQ` | `IndexIVFPQ` | 产品量化压缩存储，进一步节省内存 |
| `HNSW` | `IndexHNSWFlat` | 分层可导航小世界图，高召回高速 |

## 关键实现细节

### 1. 相似度度量

使用**内积**替代余弦相似度，因为向量在添加前已归一化：

```python
self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product
```

```python
faiss.normalize_L2(embeddings)  # L2 归一化
```

归一化后的内积 = 余弦相似度

### 2. IVF 索引构建流程

```python
quantizer = faiss.IndexFlatIP(self.embedding_dim)  # 质心计算器
self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)  # nlist=聚类数
self.index.train(embeddings)   # 先训练（聚类）
self.index.add(embeddings)     # 再添加向量
```

### 3. 搜索时归一化

```python
query_embeddings = query_embeddings.astype('float32')
faiss.normalize_L2(query_embeddings)
scores, indices = self.index.search(query_embeddings, top_k)
```

### 4. 持久化

- `save()`: `faiss.write_index()` 写入磁盘
- `load()`: `faiss.read_index()` 从磁盘读取

## 使用场景

- **开发/测试**：用 `FlatIndex`（简单可调试）
- **生产环境**：根据数据规模选择 `IVF`（中等规模）或 `HNSW`（大规模高速）
