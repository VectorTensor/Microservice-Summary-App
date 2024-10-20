from nameko.rpc import rpc
from utils.ZeroShot import ZeroShotInference
from utils.Inference import OneShot, FewShot
from dataset import load_dataset


class InContextService:
    name = "incontext"

    def __init__(self):
        self.zeroShot = ZeroShotInference("google/flan-t5-base")
        huggingface_dataset_name = "knkarthick/dialogsum"
        dataset = load_dataset(huggingface_dataset_name)
        self.oneShot = OneShot(dataset, "google/flan-t5-base")
        self.fewShot = FewShot(dataset, "google/flan-t5-base")

    @rpc
    def generate(self, input_text):
        output = self.zeroShot.generate(input_text)
        # output = "hello everyone"
        return output
