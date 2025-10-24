from together import Together

client = Together()
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    messages=[
      {
        "role": "system",
        "content": "Capitalize the first letter of the word you are given"
      },
      {
        "role": "user",
        "content": "ghost"
      }
    ]
)

print('Assistant:')
print(response.choices[0].message.content)