## for data
import pandas as pd
import numpy as np
import re
from datetime import datetime
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for machine learning
from sklearn import metrics, preprocessing
## for deep learning
from keras import models, layers, utils  #(2.6.0)


if __name__ == '__main__':
    dtf_products = pd.read_excel("data_movies.xlsx", sheet_name="products")
    dtf_users = pd.read_excel("data_movies.xlsx", sheet_name="users").head(10000)
# Products
    dtf_products = dtf_products[~dtf_products["genres"].isna()]
    dtf_products["product"] = range(0,len(dtf_products))
    dtf_products["name"] = dtf_products["title"].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x).strip())
    dtf_products["date"] = dtf_products["title"].apply(lambda x: int(x.split("(")[-1].replace(")","").strip())
    if "(" in x else np.nan)
    dtf_products["date"] = dtf_products["date"].fillna(9999)
    dtf_products["old"] = dtf_products["date"].apply(lambda x: 1 if x < 2000 else 0)

    # Users
    dtf_users["user"] = dtf_users["userId"].apply(lambda x: x-1)
    dtf_users["timestamp"] = dtf_users["timestamp"].apply(lambda x: datetime.fromtimestamp(x))
    dtf_users["daytime"] = dtf_users["timestamp"].apply(lambda x: 1 if 6<int(x.strftime("%H"))<20 else 0)
    dtf_users["weekend"] = dtf_users["timestamp"].apply(lambda x: 1 if x.weekday() in [5,6] else 0)
    dtf_users = dtf_users.merge(dtf_products[["movieId","product"]], how="left")
    dtf_users = dtf_users.rename(columns={"rating":"y"})

    # Clean
    dtf_products = dtf_products[["product","name","old","genres"]].set_index("product")
    dtf_users = dtf_users[["user","product","daytime","weekend","y"]]

    dtf_context = dtf_users[["user","product","daytime","weekend"]]

    tags = [i.split("|") for i in dtf_products["genres"].unique()]
    columns = list(set([i for lst in tags for i in lst]))
    columns.remove('(no genres listed)')
    for col in columns:
        dtf_products[col] = dtf_products["genres"].apply(lambda x: 1 if col in x else 0)

    fig, ax = plt.subplots(figsize=(20,5))
    sns.heatmap(dtf_products==0, vmin=0, vmax=1, cbar=False, ax=ax).set_title("Products x Features")
    plt.show()

    tmp = dtf_users.copy()
    dtf_users = tmp.pivot_table(index="user", columns="product", values="y")
    missing_cols = list(set(dtf_products.index) - set(dtf_users.columns))
    for col in missing_cols:
        dtf_users[col] = np.nan
    dtf_users = dtf_users[sorted(dtf_users.columns)]

    dtf_users = pd.DataFrame(preprocessing.MinMaxScaler(feature_range=(0.5,1)).fit_transform(dtf_users.values),
    columns=dtf_users.columns, index=dtf_users.index)

    split = int(0.8*dtf_users.shape[1])
    dtf_train = dtf_users.loc[:, :split-1]
    dtf_test = dtf_users.loc[:, split:]

    train = dtf_train.stack(dropna=True).reset_index().rename(columns={0:"y"})
    train.head()
    embeddings_size = 50
    usr, prd = dtf_users.shape[0], dtf_users.shape[1]

    # Users (1,embedding_size)
    xusers_in = layers.Input(name="xusers_in", shape=(1,))
    xusers_emb = layers.Embedding(name="xusers_emb", input_dim=usr, output_dim=embeddings_size)(xusers_in)
    xusers = layers.Reshape(name='xusers', target_shape=(embeddings_size,))(xusers_emb)

    # Products (1,embedding_size)
    xproducts_in = layers.Input(name="xproducts_in", shape=(1,))
    xproducts_emb = layers.Embedding(name="xproducts_emb", input_dim=prd, output_dim=embeddings_size)(xproducts_in)
    xproducts = layers.Reshape(name='xproducts', target_shape=(embeddings_size,))(xproducts_emb)

    # Product (1)
    xx = layers.Dot(name='xx', normalize=True, axes=1)([xusers, xproducts])

    # Predict ratings (1)
    y_out = layers.Dense(name="y_out", units=1, activation='linear')(xx)

    # Compile
    model = models.Model(inputs=[xusers_in,xproducts_in], outputs=y_out, name="CollaborativeFiltering")
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_percentage_error'])

#    utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    # Train
    training = model.fit(x=[train["user"], train["product"]], y=train["y"], epochs=100, batch_size=128, shuffle=True, verbose=0, validation_split=0.3)
    model = training.model
    # Test
    dtf_test = dtf_users.loc[:, split:]
    print("non-null data:", dtf_test[dtf_test > 0].count().sum())
    dtf_test

    embeddings_size = 50
    usr, prd = dtf_users.shape[0], dtf_users.shape[1]
    # Input layer
    xusers_in = layers.Input(name="xusers_in", shape=(1,))
    xproducts_in = layers.Input(name="xproducts_in", shape=(1,))

    # A) Matrix Factorization
    ## embeddings and reshape
    cf_xusers_emb = layers.Embedding(name="cf_xusers_emb", input_dim=usr, output_dim=embeddings_size)(xusers_in)
    cf_xusers = layers.Reshape(name='cf_xusers', target_shape=(embeddings_size,))(cf_xusers_emb)
    ## embeddings and reshape
    cf_xproducts_emb = layers.Embedding(name="cf_xproducts_emb", input_dim=prd, output_dim=embeddings_size)(xproducts_in)
    cf_xproducts = layers.Reshape(name='cf_xproducts', target_shape=(embeddings_size,))(cf_xproducts_emb)
    ## product
    cf_xx = layers.Dot(name='cf_xx', normalize=True, axes=1)([cf_xusers, cf_xproducts])

    # B) Neural Network
    ## embeddings and reshape
    nn_xusers_emb = layers.Embedding(name="nn_xusers_emb", input_dim=usr, output_dim=embeddings_size)(xusers_in)
    nn_xusers = layers.Reshape(name='nn_xusers', target_shape=(embeddings_size,))(nn_xusers_emb)
    ## embeddings and reshape
    nn_xproducts_emb = layers.Embedding(name="nn_xproducts_emb", input_dim=prd, output_dim=embeddings_size)(xproducts_in)
    nn_xproducts = layers.Reshape(name='nn_xproducts', target_shape=(embeddings_size,))(nn_xproducts_emb)
    ## concat and dense
    nn_xx = layers.Concatenate()([nn_xusers, nn_xproducts])
    nn_xx = layers.Dense(name="nn_xx", units=int(embeddings_size/2), activation='relu')(nn_xx)

    # Merge A & B
    y_out = layers.Concatenate()([cf_xx, nn_xx])
    y_out = layers.Dense(name="y_out", units=1, activation='linear')(y_out)
    # Compile
    model = models.Model(inputs=[xusers_in,xproducts_in], outputs=y_out, name="Neural_CollaborativeFiltering")
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_percentage_error'])

    model.summary()

