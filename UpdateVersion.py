with open("VERSION", mode="r+") as f:
  version = f.read().split(".")
  version[-1] = str(int(version[-1]) + 1)
  f.seek(0)
  f.write(".".join(version))