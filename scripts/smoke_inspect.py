import autoresearch_core as arc

print("dir:", dir(arc))
print("hasattr pareto:", hasattr(arc, "pareto"))
print("sys.modules:", [k for k in __import__("sys").modules if "autoresearch" in k])
