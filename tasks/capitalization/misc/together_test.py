from together import Together

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)

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

logger.info('Assistant:')
logger.info(response.choices[0].message.content)