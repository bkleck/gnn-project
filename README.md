# gnn-project
*Graph Neural Network (GNN) for Smart Investment Module*
<br/> 

<img src='https://user-images.githubusercontent.com/77097236/126464780-614e8723-2f8d-41fb-a8d6-31d5d8eb9b20.jpg' width='500' height='300'>
<br/> 

## Table of Contents
* [Introduction](#introduction)
* [Technology](#technology)
* [Documentation](#documentation)
  * [Data Processing](#1-data-processing)
  * [Document Embeddings](#2-create-document-embeddings)
  * [Knowledge Graph](#3-create-knowledge-graph)
  * [Graph Neural Network](#4-graph-neural-network)
* [Possible Improvements](#possible-improvements)

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

Finally, we will add in [attributes](https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.set_node_attributes.html) into our empty nodes. All of our **_text data will be transformed into a 1D tensor_** and set as a **_'feature' attribute_**, to be used for training of the GNN. The allocation of the companies **_(active, watchlight, not interested)_** will be set as an **_'allocation' attribute_**, to be used as labels for the GNN output.
<br/> 
<br/> 

### 4) Graph Neural Network
As mentioned earlier, we will be using StellarGraph for GNN training and prediction. This is because it supports loading data in the format of a **_NetworkX object_**, and it also contains many of the **_state-of-the-art graph neural models_**.

In our graph, our tensors will be used as features for training, while the allocations will be what we are trying to predict. This helps us determine which companies we should be looking more to invest in.

Now, we will load in the graph and **_set the 'features' attribute (text data) as node features_**.

```
stellar = StellarGraph.from_networkx(G, node_features='feature')
```
<br/> 

I intentionally made the nodes of all type 'default' as many common neural networks like GCN, GAT only support **_single-node-type_**. I also created indices for the graph model to use as a reference for **_splitting into training, validation and test datasets_**. 

For company nodes, we will label them according to 'active', 'watchlight' or not 'interested'. For venture, industry and technology nodes, we will label them as 'default'.

We will be using the **_LabelBinarizer_** to encode the node names into unique numbers so that they can be input into the graph.
<br/> 
<br/> 

First model we will be trying out is the [Graph Convolutional Network](https://towardsdatascience.com/understanding-graph-convolutional-networks-for-node-classification-a2bfdb7aba7b). The GCN class packages the **_stack of convolutional layers and dropout layers_**, with layer size = (number of hidden GCN layers, size of each layer). The **_generator converts graph data into Keras format_**.
```
generator = FullBatchNodeGenerator(stellar, method="gcn")
gcn = GCN(layer_sizes=[16, 16], activations=["relu", "relu"], generator=generator, dropout=0.2)
```
*Results obtained:*

![image](https://user-images.githubusercontent.com/77097236/126605823-bff5a976-908b-4c35-9754-a61a9907f69b.png)
<br/> 
<br/> 

Next model we will be trying out is the [Graph Attention Network](https://paperswithcode.com/method/gat), an improved GNN which utilizes the **_'Attention' mechanism_**, similar to the BERT-series.
```
generator = FullBatchNodeGenerator(stellar, method="gat")
gat = GAT(
    layer_sizes=[8, train_targets.shape[1]],
    activations=["elu", "softmax"],
    attn_heads=8,
    generator=generator,
    in_dropout=0.5,
    attn_dropout=0.5,
    normalize=None,
)
```

*Results obtained:*

![image](https://user-images.githubusercontent.com/77097236/126606891-7f91c0c5-13ad-44ff-9e59-0b339b64383e.png)
<br/> 
<br/> 

## Possible Improvements
1) Add in freshly encoded interesy/urgency scores to create more training data with both January and May datasets.
2) Make use of GNNs that support multi node-types to ignore nodes like ventures, industries and technologies during training.
3) Add in summaries over time to provide more features for training.
