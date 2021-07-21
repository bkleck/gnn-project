# gnn-project
*Graph Neural Network (GNN) for Smart Investment Module*
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/126464780-614e8723-2f8d-41fb-a8d6-31d5d8eb9b20.jpg' width='500' height='300'>
<br/> 

## Table of Contents
* [Introduction](#introduction)
* [Technology](#technology)
* [Workflow](#workflow)
  * [Cleaning Scraped News](#1-cleaning-scraped-news)
* [File Descriptions](#file-descriptions)
* [Setup](#setup)
* [How to Use](#how-to-use)
* [Improvements](#improvements)
* [Data Flow](#data-flow)

## Introduction
With the large source of data collected from news and reports, as well as from our own database of companies, these data can be put together to **_predict the interest and urgency scores_** for investing in portfolio companies. 

Rather than making this a conventional classification problem, we have decided to **_utilize graph neural networks for node classification_** instead. This will help make better predictions as graph networks take into account connections between nodes to make more informed decisions based on similarities between clusters of nodes.
<br/> 

## Technology
- [HuggingFace](https://huggingface.co/sentence-transformers)  
This python library was used to encode our text data into tensors for input into algorithms.  
- [NetworkX](https://networkx.org/)  
This library helps us to generate a knowledge graph from our data points.  
- [StellarGraph](https://stellargraph.readthedocs.io/en/stable/)  
This library was used to perform deep learning on our graph object. We chose this library as it has the most documentation on GNNs, and also supports loading our data as a NetworkX object which provides a lot of convenience. 
<br/> 

## Documentation
### 1) Data Processing
This step is performed in the `clean_data.ipynb` file.
Firstly, we will be pulling **_data labelled by the Investment Team_** and merging them together. 

Afterwards, we will be pulling additional information from the company's main database, such as **_industry, technology and basic description_** of each company.
These data, together with the companies, will serve as **_nodes_** within the graph.

We will also be **_cleaning out symbols and noise in the text data_** before passing them into the SentenceTransformer later on. Furthermore, **_scores for January and May 2021 will be re-encoded_** to match each other.

Within the file, the **_interest & urgency score distributions_** have been plotted to look for anomalies that could deteriorate model training. 
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/126469480-96cc8d48-d0bd-40dc-8e34-1690bb497171.png' width='350' height='250' alight='left'>

> One problem identified was the uneven distribution for May urgency scores. However, as this is tied to the fundraising cycle, it does not require modification and helps to add good variance to the data.
<br/> 

### 2) Create Document Embeddings
From this step onwards, they are all performed in the `create_graph.ipynb` file.

To feed our text data into the graph neural network, we will need to encode it in a way that the model will understand. Hence, we will be using BERT to **_encode the paragraphs of text into tensors_** for embedding. This will output a tensor of size 768. These tensors will then be used later on as **_features in our graph for training_** of our graph neural network.

We will make use of the **_SentenceTransformer from HuggingFace_** to encode the text data. Each paragraph will be passed in as a Sentence object to encode them all into **_standard BERT-sized tensors_**.

![image](https://user-images.githubusercontent.com/77097236/126471719-5b15e72c-4008-4339-bf83-ccfbdd522e34.png)
<br/> 
<br/> 

### 3) Create Knowledge Graph
We will be using NetworkX to create our knowledge graph. NetworkX allows us to create nodes and edges in the form of triples: **_subject-edge-object_**.

![image](https://user-images.githubusercontent.com/77097236/126472377-ae6f1569-2842-40cc-a63b-6e226ed9b521.png)

This actually provides much **_better ease of implementation_** as compared to many other libraries, such as Pytorch Geometric, which only allow creation of graph through indexing.

Hence, we will make use of the **_graph_data_** function to convert our dataframe into a similar format, by connecting each of the companies with their venture, industry and technologies.
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/126473792-14489967-e3ae-4aed-a8d6-3eaa578d4d52.png' height='200' width='400'>

Now, we have successfully converted our dataframe into meaningful nodes for input into the graph.
The **_companies_** will be the **_source nodes_**, with the **_target nodes_** being the **_ventures, industries and technologies_**. These nodes will be joined by **_edges with the labels 'attributes'_**.

The graph is created with this line of code:
```
G = nx.from_pandas_edgelist(graph_df, 'source','target',['attribute'])
```

This is an example of how our network graph will look like:
![050721_graph](https://user-images.githubusercontent.com/77097236/126474897-18cafbd3-ed46-4e83-ba20-5cd3e545723e.jpg)

Legend:
- **Orange = Ventures**
- **Purple = Industries**
- **Pink = Tech**
- **Turquoise = Companies**
