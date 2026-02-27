import argparse
import dspy

parser = argparse.ArgumentParser(description="Simple DSPy capitalizer demo.")
parser.add_argument('--model_path', type=str, required=True, help='Path to the model directory')
args = parser.parse_args()

lm = dspy.LM(model=args.model_path)
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
    print(result.capitalized_word)
