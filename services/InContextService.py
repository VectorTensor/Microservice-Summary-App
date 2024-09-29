from nameko.rpc import rpc
from utils.ZeroShot import ZeroShotInference


class InContextService:
    name = "incontext"

    def __init__(self):
        self.zeroShot = ZeroShotInference("google/flan-t5-base")

    @rpc
    def generate(self, input_text):
        output = self.zeroShot.generate(input_text)
        # output = "hello everyone"
        return output
