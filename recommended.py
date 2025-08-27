import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

#  Veri Hazırlığı

styles_df = pd.read_csv('styles.csv', on_bad_lines='skip')
df_gorsellestir = styles_df.copy()

# Eksik değerler
styles_df['productDisplayName'] = styles_df['productDisplayName'].fillna('')
label_encoder = LabelEncoder()
styles_df['baseColour'] = label_encoder.fit_transform(styles_df['baseColour'].astype(str))
styles_df['season'] = label_encoder.fit_transform(styles_df['season'].astype(str))
imputer = KNNImputer(n_neighbors=5)
styles_df[['baseColour', 'season']] = imputer.fit_transform(styles_df[['baseColour', 'season']])

# One-hot encoding
one_hot_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season']
encoder = OneHotEncoder(sparse_output=False, drop='first')
one_hot_encoded = encoder.fit_transform(styles_df[one_hot_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(one_hot_columns))
styles_df = styles_df.drop(columns=one_hot_columns)
styles_df = pd.concat([styles_df, one_hot_df], axis=1)

# TF-IDF 
vectorizer = TfidfVectorizer(max_features=1000)
text_features = vectorizer.fit_transform(styles_df['productDisplayName'].fillna('')).toarray()
styles_df = styles_df.drop(columns=['productDisplayName'])
text_df = pd.DataFrame(text_features, columns=[f'productDisplayName_{i}' for i in range(text_features.shape[1])])
styles_df = pd.concat([styles_df, text_df], axis=1)

# Usage label
y = pd.get_dummies(styles_df["usage"], drop_first=True).astype(int)
styles_df = pd.concat([styles_df, y], axis=1)
styles_df.drop(["usage"], axis=1, inplace=True)


# Görsellerin Yüklenmesi

image_dir = 'images/'
def load_images(image_dir, image_ids, target_size=(64,64)):
    images = []
    counter = 0
    for img_id in image_ids:
        img_path = f"{image_dir}/{img_id}"
        try:
            img = image.load_img(img_path, target_size=target_size)
            img_array = image.img_to_array(img)
            images.append(img_array)
        except FileNotFoundError:
            counter += 1
            print(f"Bulunamayan gorsel sayisi: {counter}")
            images.append(np.zeros((target_size[0], target_size[1],3)))
    return np.array(images)

image_ids = styles_df['id'].astype(str) + '.jpg'
X_images = load_images(image_dir, image_ids)
X_images = X_images / 255.0
X_tabular = styles_df.drop(columns=['id']).values

#  Embedding Modeli (Görsel + Tabular + Text)

def build_embedding_model(input_shape_image=(64,64,3), tabular_dim=X_tabular.shape[1]):
    # Görsel
    image_input = Input(shape=input_shape_image, name='image_input')
    x = Conv2D(32,(3,3),activation='relu')(image_input)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    image_features = Dense(64, activation='relu')(x)
    
    
    tabular_input = Input(shape=(tabular_dim,), name='tabular_input')
    y = Dense(64, activation='relu')(tabular_input)
    y = Dense(32, activation='relu')(y)
    y = Dropout(0.3)(y)
    tabular_features = Dense(16, activation='relu')(y)
    
    
    merged = Concatenate()([image_features, tabular_features])
    embedding = Dense(128, activation='relu', name='embedding')(merged)
    
    model = Model(inputs=[image_input, tabular_input], outputs=embedding)
    return model

embedding_model = build_embedding_model()
embedding_model.summary()

# Triplet Loss

def triplet_loss(margin=0.3):
    def loss(y_true, y_pred):
        anchor, positive, negative = y_pred[:,0], y_pred[:,1], y_pred[:,2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss

optimizer = Adam(learning_rate=1e-3)
embedding_model.compile(optimizer=optimizer, loss='mse')  

# Triplet Veri Üretici

def generate_triplets(X_img, X_tab, usage_labels):
    anchors_img, anchors_tab = [], []
    positives_img, positives_tab = [], []
    negatives_img, negatives_tab = [], []
    
    n = len(X_img)
    for i in range(n):
        anchor_img, anchor_tab, anchor_label = X_img[i], X_tab[i], usage_labels[i]
        
        # Positive
        pos_idx = np.random.choice(np.where(usage_labels==anchor_label)[0])
        positive_img, positive_tab = X_img[pos_idx], X_tab[pos_idx]
        
        # Negative
        neg_idx = np.random.choice(np.where(usage_labels!=anchor_label)[0])
        negative_img, negative_tab = X_img[neg_idx], X_tab[neg_idx]
        
        anchors_img.append(anchor_img)
        anchors_tab.append(anchor_tab)
        positives_img.append(positive_img)
        positives_tab.append(positive_tab)
        negatives_img.append(negative_img)
        negatives_tab.append(negative_tab)
    
    
    return (np.array(anchors_img), np.array(anchors_tab),
            np.array(positives_img), np.array(positives_tab),
            np.array(negatives_img), np.array(negatives_tab))

usage_labels = np.argmax(y.values, axis=1)  
anchors_img, anchors_tab, positives_img, positives_tab, negatives_img, negatives_tab = \
    generate_triplets(X_images, X_tabular, usage_labels)

# Embedding Çıkarma ve Öneri Sistemi

embeddings = embedding_model.predict([X_images, X_tabular])

def rec(liked_indices, embeddings, top_n=5):
    liked_features = embeddings[liked_indices]
    last_liked_feature = liked_features[-1]
    last_similarity = cosine_similarity([last_liked_feature], embeddings).flatten()
    last_recommendations = np.argsort(last_similarity)[::-1][:top_n]
    
    second_last_feature = liked_features[-2]
    second_last_similarity = cosine_similarity([second_last_feature], embeddings).flatten()
    second_last_recommendations = np.argsort(second_last_similarity)[::-1][:top_n-2]
    
    first_liked_feature = liked_features[0]
    first_similarity = cosine_similarity([first_liked_feature], embeddings).flatten()
    first_recommendations = np.argsort(first_similarity)[::-1][:top_n-3]
    
    all_recommendations = np.concatenate([last_recommendations, second_last_recommendations, first_recommendations])
    unique_recommendations = np.unique(all_recommendations)
    unique_recommendations = [idx for idx in unique_recommendations if idx not in liked_indices]
    return unique_recommendations


liked_indices = [11,22,35]
unique_recommendations = rec(liked_indices, embeddings)

plt.figure(figsize=(15,6))
for i, idx in enumerate(unique_recommendations[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(X_images[idx])
    plt.title(f"Index: {idx}")
    plt.axis('off')
plt.show()
