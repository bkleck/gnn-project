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

## Technology
- [HuggingFace](https://huggingface.co/sentence-transformers)  
This python library was used to encode our text data into tensors for input into algorithms.  
- [NetworkX](https://networkx.org/)  
This library helps us to generate a knowledge graph from our data points.  
- [StellarGraph](https://stellargraph.readthedocs.io/en/stable/)  
This library was used to perform deep learning on our graph object. We chose this library as it has the most documentation on GNNs, and also supports loading our data as a NetworkX object which provides a lot of convenience. 

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

### 2) Create Knowledge Graph

