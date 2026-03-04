import pandas as pd
dfjo = pd.DataFrame(
    dict(A=range(1,4), B=range(4,7), C=range(7, 10)),
    columns=["A", "B", "C"],
    index=list("xuz"),
)

print(dfjo)

dfjo.to_json("df.json", orient="records")

