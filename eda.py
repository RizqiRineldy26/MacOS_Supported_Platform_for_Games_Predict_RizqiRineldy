import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

def run():
    st.title('MacOS compatible platform for games prediction')

    #Tampilkan gambar
    # img = Image.open('bola.png')
    # st.image(img, caption = 'FIFA 2022')

    #load data
    df1 = pd.read_csv('games.csv')
    st.dataframe(df1)

    #membuat barplot
    st.write('### Distribusi Target')
    target_counts = df1['mac'].value_counts()
    target_props = df1['mac'].value_counts(normalize=True)

    print(target_counts)
    print(target_props)

    fig, ax = plt.subplots(figsize=(5, 4))
    target_counts.plot(kind='bar', color=['#4C72B0', '#DD8452'], ax=ax)
    ax.set_title('Distribusi Target: mac')
    ax.set_xlabel('mac')
    ax.set_ylabel('Jumlah game')

    for i, v in enumerate(target_counts):
        ax.text(i, v + 500, str(v), ha='center')

    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    # membuat barplot kedua
    st.write('### Review Games terhadap Harga')
    bins = [0, 5, 10, 20, 50, df1['price'].max()]
    labels = ['0-5', '5-10', '10-20', '20-50', '50+']
    df1['price_bin'] = pd.cut(df1['price'], bins=bins, labels=labels, include_lowest=True)

    review_by_price = df1.groupby('price_bin', observed=True)[['positive', 'negative']].mean()

    print(review_by_price)

    fig, ax = plt.subplots(figsize=(8, 5))
    review_by_price.plot(kind='bar', ax=ax, color=['#55A868', '#C44E52'])
    ax.set_title('Rata-rata Review Positif & Negatif berdasarkan Rentang Harga')
    ax.set_xlabel('Rentang Harga (price)')
    ax.set_ylabel('Rata-rata jumlah review')
    ax.legend(['positive', 'negative'])
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

if __name__ == "__main__":
    run()