

# import datasets

# all_ds = datasets.list_datasets()
# len(all_ds)

# all_ds[:5]





# from transformers import whoami





import requests                                                                                                                                                                                                         
result = requests.head("https://raw.githubusercontent.com/huggingface/datasets/1.1.2/datasets/cnn_dailymail/cnn_dailymail.py")
print(result)


