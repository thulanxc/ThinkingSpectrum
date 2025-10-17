# 导入 openai 库
import openai

# 创建一个 OpenAI 客户端实例
# 注意：我们需要在这里指定您的 API URL 和 API Key
# 1. base_url: 指向您提供的 API 地址 "https://api.laozhang.ai/v1"
# 2. api_key: 填入您提供的 API Key
client = openai.OpenAI(
    base_url="https://api.moonshot.cn/v1",
    api_key="sk-RXvmowS7kvaEcPpH6E2u4soQE5PZ1N02ZnUALsEscG60uygF",
)

# 准备请求的数据
# model: 您指定的模型名称
# messages: 您要发送给模型的内容
# max_tokens: 指定模型生成内容的最大长度
data = {
    "model": "kimi-k2-turbo-preview",
    "max_tokens": 4096,
    "messages": [
        {
            "role": "user",
            "content": "Ответьте по-английски, какая вы модель?"
        }
    ],
    "temperature": 0.6,
}

# 使用 try...except 结构来捕获可能发生的网络或 API 错误
try:
    # 调用 chat.completions.create 方法来发送请求
    # 使用 **data 将字典中的所有键值对作为参数传递给该方法
    completion = client.chat.completions.create(**data)

    # 如果请求成功，打印出模型返回的内容
    print("Result：")
    print(completion.choices[0].message.content)

except Exception as e:
    # 如果发生错误，打印出错误信息
    print(f"请求出错了，错误信息：{e}")