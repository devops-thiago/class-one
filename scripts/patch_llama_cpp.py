import os

temp_dir = os.environ.get("TEMP")
gemma_path = os.path.join(temp_dir, "llama_cpp", "conversion", "gemma.py")

with open(gemma_path, encoding="utf-8") as f:
    c = f.read()

target = """    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item"""

replacement = """    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if not name.startswith("model.") and not name.startswith("lm_head."):
            name = "model." + name"""

if target in c:
    c = c.replace(target, replacement)
    with open(gemma_path, "w", encoding="utf-8") as f:
        f.write(c)
    print("Updated filter_tensors in Gemma4Model")
else:
    print("Target not found or already modified")
