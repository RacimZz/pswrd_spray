import pandas as pd

df = pd.read_csv("data/public/rba/rba.csv")
print(f"Total : {len(df)} lignes")
print(f"Attaques : {df['Is Attack IP'].sum()} ({df['Is Attack IP'].mean():.2%})")

sample = df.groupby("Is Attack IP", group_keys=False).apply(
    lambda x: x.sample(min(len(x), int(1_000_000 * len(x) / len(df))), random_state=42)
)

print(f"Sample : {len(sample)} lignes")
print(f"Attaques dans sample : {sample['Is Attack IP'].sum()} ({sample['Is Attack IP'].mean():.2%})")

sample.to_csv("data/public/rba/rba_1m.csv", index=False)
print("Sauvegarde : data/public/rba/rba_1m.csv")
