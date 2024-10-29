import wikipediaapi
#conda 安装环境 conda install conda-forge::wikipedia-api


def augment_data(df):
    # Define a custom user-agent
    USER_AGENT = ""    # 改一下自己的

    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=USER_AGENT
    )

    def get_wikipedia_summary(query):
        """Fetches a short summary from Wikipedia given a query."""
        try:
            page = wiki_wiki.page(query)
            if page.exists():
                # Get a short summary (truncate to about 200 characters)
                summary = page.summary[:200]
                return summary
            else:
                return ""
        except Exception as e:
            print(f"Error retrieving Wikipedia page: {e}")
            return ""

    # Augment each prompt with Wikipedia context
    for i, prompt in enumerate(df.prompt.values):
        wiki_summary = get_wikipedia_summary(prompt.split()[0])  # Modify to choose appropriate keyword
        if wiki_summary:
            # Append to the prompt
            df.at[i, 'prompt'] = f"{prompt} ? Context: {wiki_summary}"

    return df




from sentence_transformers import SentenceTransformer, util
# conda install conda-forge::sentence-transformers 安装命令
import torch


def RAG_FROM_WIKI(test_df):

    # 直接把train_df送进来 , 即为 (1, 6) 的二维数组 1个测试问题 6列分别为 index question A B C D
    # 这里只用比较 A B C D 四个选项与wiki检索prompt文本的匹配度即可

    model = SentenceTransformer('all-MiniLM-L6-v2')  # 用此模型来编码句子并计算相似度

    df = augment_data(test_df)# 生成的wiki检索结果


    sentence_embeddings_A = model.encode(df['A'].values)  # 原始选项A编码
    sentence_embeddings_B = model.encode(df['B'].values)  # 原始选项B编码
    sentence_embeddings_C = model.encode(df['C'].values)  # 原始选项C编码
    sentence_embeddings_D = model.encode(df['D'].values)  # 原始选项D编码
    sentence_embeddings_E = model.encode(df['E'].values)  # 原始选项D编码
    context_embeddings = model.encode(df['prompt'].apply(lambda x: x.split("Context: ")[-1]).values)  # 提取摘要编码

    # 计算余弦相似度
    similarity_scores_A = util.cos_sim(sentence_embeddings_A, context_embeddings)
    similarity_scores_B = util.cos_sim(sentence_embeddings_B, context_embeddings)
    similarity_scores_C = util.cos_sim(sentence_embeddings_C, context_embeddings)
    similarity_scores_D = util.cos_sim(sentence_embeddings_D, context_embeddings)
    similarity_scores_E = util.cos_sim(sentence_embeddings_E, context_embeddings)

    similarity_scores = [
        ("A", similarity_scores_A),
        ("B", similarity_scores_B),
        ("C", similarity_scores_C),
        ("D", similarity_scores_D),
        ("E", similarity_scores_E),
    ]

    # 按照相似度分数从大到小排序
    # 注意：如果 similarity_scores 是多维的（例如每个选项有多个分数），可以用 torch.max() 或 .max() 提取最大值
    sorted_scores = sorted(similarity_scores,
                           key=lambda x: x[1].max().item() if isinstance(x[1], torch.Tensor) else max(x[1]),
                           reverse=True)

    # 返回前三个大的排序结果
    top_3 = sorted_scores[:3]
    return top_3


#    for i in range(test_df.shap[0]):
#        test_df_i = test_df.loc[i, ['A', 'B', 'C', 'D']] # 一行一行提取比对
#        for j in range(4):
#            test_df_1 = test_df_i.iloc[:, j] # 一个一个提取比对
#           df = augment_data(test_df_1)

#            sentence_embeddings = model.encode(df.values)  # 原始句子编码
#           context_embeddings = model.encode(df.apply(lambda x: x.split("Context: ")[-1]).values)  # 提取摘要编码





