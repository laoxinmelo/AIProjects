# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
import chromadb
import os

# 设置 API Key
key = os.environ.get("DASHSCOPE_API_KEY")


def llm_test():
    llm = ChatTongyi(model="qwen-max")  # ✅ 关键字参数
    response = llm.invoke("用一句话解释什么是 RAG")
    print(response.content)

def embedding_test():
    print("\n" + "=" * 50)
    print("测试 Embedding")
    print("=" * 50)
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    text = "AC米兰是意大利足球俱乐部"
    vector = embeddings.embed_query(text)
    print(f"文本：{text}")
    print(f"向量维度：{len(vector)}")
    print(f"前 5 个值：{vector[:5]}")

def chromaDB_test():
    print("\n" + "=" * 50)
    print("测试 ChromaDB 向量存储")
    print("=" * 50)
    client = chromadb.Client()
    collection = client.create_collection("test")
    documents = ["AC 米兰", "国际米兰", "尤文图斯"]
    ids = ["1", "2", "3"]
    # 生成嵌入向量
    # 生成嵌入向量
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")
    doc_embeddings = embeddings.embed_documents(documents)

    # 添加到集合
    collection.add(
        ids=ids,
        embeddings=doc_embeddings,
        documents=documents  # 建议同时保存原文
    )
    print(f"成功添加 {len(documents)} 个文档到集合中")

    results = collection.query(
        query_embeddings=embeddings.embed_query("米兰"),
        n_results=2
    )
    print(f"\n查询'米兰'的结果：{results['documents'][0]}")

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    llm_test()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
