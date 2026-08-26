import streamlit as st
import pickle
import pandas as pd
import datetime

with open('model_gb.pkl', 'rb') as file_1:
    model_gb = pickle.load(file_1)


def run():
    st.title('Steam Game - macOS Support Prediction')
    st.write('Isi detail game di bawah, lalu klik **Predict** untuk memprediksi '
             'apakah game tersebut kemungkinan mendukung macOS.')

    with st.form("steam_form"):
        st.subheader('Informasi Umum')
        name = st.text_input('Nama Game', value='Counter-Strike 2')
        release_date = st.date_input('Tanggal Rilis', value=datetime.date(2012, 8, 21))
        required_age = st.number_input('Required Age', min_value=0, max_value=21, value=0)
        price = st.number_input('Harga (USD)', min_value=0.0, max_value=300.0, value=0.0, step=0.5)
        dlc_count = st.number_input('Jumlah DLC', min_value=0, max_value=1000, value=1)
        discount = st.slider('Discount (%)', 0, 100, 0)

        st.subheader('Platform')
        windows = st.selectbox('Windows', (True, False), index=0)
        linux = st.selectbox('Linux', (True, False), index=0)

        st.subheader('Rilis & Kredit')
        developers = st.text_input('Developer', value='Valve')
        publishers = st.text_input('Publisher', value='Valve')
        categories = st.text_area('Categories (pisahkan dengan koma)',
                                   value='Multi-player, Cross-Platform Multiplayer, Steam Trading Cards')
        genres = st.text_area('Genres (pisahkan dengan koma)', value='Action, Free To Play')
        supported_languages = st.text_area('Supported Languages (pisahkan dengan koma)',
                                            value='English, Indonesian')
        full_audio_languages = st.text_area('Full Audio Languages (pisahkan dengan koma)',
                                             value='English, Indonesian')

        st.subheader('Skor & Review')
        metacritic_score = st.slider('Metacritic Score', 0, 100, 0)
        user_score = st.slider('User Score', 0, 100, 0)
        achievements = st.number_input('Achievements', min_value=0, max_value=10000, value=1)
        recommendations = st.number_input('Recommendations', min_value=0, value=0)
        positive = st.number_input('Positive Reviews', min_value=0, value=0)
        negative = st.number_input('Negative Reviews', min_value=0, value=0)
        pct_pos_total = st.slider('Persentase Review Positif (Total, %)', 0, 100, 50)
        num_reviews_total = st.number_input('Jumlah Review (Total)', min_value=0, value=0)
        pct_pos_recent = st.slider('Persentase Review Positif (Recent, %)', 0, 100, 50)
        num_reviews_recent = st.number_input('Jumlah Review (Recent)', min_value=0, value=0)

        st.subheader('Popularitas')
        estimated_owners = st.selectbox(
            'Estimated Owners',
            ('0 - 20000', '20000 - 50000', '50000 - 100000', '100000 - 200000',
             '200000 - 500000', '500000 - 1000000', '1000000 - 2000000',
             '2000000 - 5000000', '5000000 - 10000000', '10000000 - 20000000',
             '20000000 - 50000000', '50000000 - 100000000', '100000000 - 200000000'),
            index=3
        )
        average_playtime_forever = st.number_input('Average Playtime Forever (menit)', min_value=0, value=0)
        peak_ccu = st.number_input('Peak CCU', min_value=0, value=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        data_inf = {
            'name': name,
            'release_date': release_date.strftime('%Y-%m-%d'),
            'required_age': required_age,
            'price': price,
            'dlc_count': dlc_count,
            'windows': windows,
            'linux': linux,
            'metacritic_score': metacritic_score,
            'achievements': achievements,
            'recommendations': recommendations,
            'developers': developers,
            'publishers': publishers,
            'categories': categories,
            'genres': genres,
            'supported_languages': supported_languages,
            'full_audio_languages': full_audio_languages,
            'user_score': user_score,
            'positive': positive,
            'negative': negative,
            'estimated_owners': estimated_owners,
            'average_playtime_forever': average_playtime_forever,
            'discount': discount,
            'peak_ccu': peak_ccu,
            'pct_pos_total': pct_pos_total,
            'num_reviews_total': num_reviews_total,
            'pct_pos_recent': pct_pos_recent,
            'num_reviews_recent': num_reviews_recent,
        }

        data_inf_df = pd.DataFrame([data_inf])

        st.write('### Data Input')
        st.dataframe(data_inf_df)

        y_pred_inf = model_gb.predict(data_inf_df)
        pred_label = 'Yes (Mendukung macOS)' if bool(y_pred_inf[0]) else 'No (Tidak Mendukung macOS)'

        st.write('## Prediksi:', pred_label)

        if hasattr(model_gb, 'predict_proba'):
            proba = model_gb.predict_proba(data_inf_df)[0]
            st.write(f'Confidence: {max(proba) * 100:.2f}%')


if __name__ == "__main__":
    run()
