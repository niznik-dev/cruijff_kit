import dspy

from cruijff_kit.utils.logger import setup_logger

# Set up logging
logger = setup_logger(__name__)

lm = dspy.LM(model="/scratch/gpfs/MSALGANIK/pretrained-llms/Llama-3.2-1B-Instruct")
dspy.settings.configure(lm=lm)

class Capitalizer(dspy.Signature):
    word = dspy.InputField()
    capitalized_word = dspy.OutputField()

class SimpleCapitalizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(Capitalizer)

    def forward(self, word):
        return self.generate(word=word)

if __name__ == "__main__":
    capitalizer = SimpleCapitalizer()
    result = capitalizer(word="apple")
    logger.info(result.capitalized_word)