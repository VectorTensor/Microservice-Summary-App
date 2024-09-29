from utils.ZeroShot import ZeroShotInference


z = ZeroShotInference("google/flan-t5-base")
print(z.generate("hello"))