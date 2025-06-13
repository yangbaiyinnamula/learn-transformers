from datasets import load_dataset

imdb = load_dataset("imdb", download_mode='force_redownload')

print(imdb['test'][0])