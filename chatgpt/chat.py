import openai

if __name__=='__main__':
    openai.api_key = "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"
    api = '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'
    prompt = "用Python写一段冒泡排序程序"
    response = openai.Completion.create(
        engine="text-davinci-003",  # 慢，模型大、能力强
        # engine="text-curie-001", # 较快
        # engine="text-babbage-001",
        # engine="text-ada-001", # 最快
        prompt=prompt,
        max_tokens=1024,  # 编码长度
        n=1,  # 候选答案数量
        temperature=1,
    )
    for answer in response["choices"]:
        print(answer["text"])