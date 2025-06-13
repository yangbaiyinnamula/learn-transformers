from transformers import pipeline 

text = "Hugging Face is a community-based open-source <mask> for machine learning."
fill_mask = pipeline(task="fill-mask")
preds = fill_mask(text, top_k=1)
preds = [
    {
        "score": round(pred["score"], 4),
        "token": pred["token"],
        "token_str": pred["token_str"],
        "sequence": pred["sequence"],
    }
    for pred in preds
]
print(preds)