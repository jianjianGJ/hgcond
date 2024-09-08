The dataset will be downloaded automatically. If the download fails, you can view the source code of `torch_geometric.datasets` and update the url.

Since the splits for datasets ACM and Freebase are randomly generated, it may lead to inconsistent results. To address the issue, you can utilize the uploaded splits instead of using on the code in utils_data.py that generates splits.

