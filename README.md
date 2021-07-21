# gnn-project
*Graph Neural Network (GNN) for Smart Investment Module*
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/126464780-614e8723-2f8d-41fb-a8d6-31d5d8eb9b20.jpg' width='500' height='300'>
<br/> 

## Table of Contents
* [Introduction](#introduction)
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
This library was used to perform deep learning on our graph object. We chose this library as it has the most documentation on GNNs, and also supports loading our data as a NetworkX object.  
