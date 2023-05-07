import streamlit as st 
import streamlit.components.v1 as stc 
from streamlit.components.v1 import iframe

# EDA Pkgs
import pandas as pd 
#import neattext as nt
import random
import numpy


# Data Viz Pkgs
import matplotlib.pyplot as plt 
from PIL import Image
import matplotlib 
matplotlib.use('Agg')
import altair as alt 
import streamlit.components.v1 as stc
import neattext.functions as nfx
import cohere
from config import cohere_key

from template import HTML_RANDOM_TEMPLATE,render_entities,get_tags,mytag_visualizer,plot_mendelhall_curve,plot_word_freq_with_altair,get_most_common_tokens


st.set_page_config(layout="centered",page_icon="📖", page_title="Bible App")

def load_bible(data): # function to load bible
	df = pd.read_csv(data)
	return df

st.title("Bible App")

menu = ["Home","MultiVerse","About"]
df = load_bible("data/KJV.csv")

co = cohere.Client(cohere_key)  # This is your trial API key

response = co.embed(
    model='large',
    texts=[" \"I like Football\"", "I like computer science and AI ", "I studied political science "])
# print('Embeddings: {}'.format(response.embeddings))
np_embedding = np.array(response.embeddings)
print(np_embedding.shape)
# np.save("co_embedding",np_embedding)




choice = st.sidebar.selectbox("Menu",menu)
if choice == "Home":
    st.subheader("Single Verse Search")
    #st.dataframe(df)
    book_list = df['book'].unique().tolist()
    book_name =st.sidebar.selectbox("Book",book_list)
    chapter = st.sidebar.number_input("Chapter",1)
    verse = st.sidebar.number_input("Verse",1)
    bible_df = df[df['book'] == book_name]
    
    
    
    image = Image.open("image/image.jpg")
    st.image(image,caption=None, width=400, use_column_width=1, clamp=False, channels="RGB", output_format="auto")



    #st.image("image.jpg")
    #st.image(image.jpg, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    # Layout
    c1,c2 = st.columns([2,1])

    # Single Verse Layout
    with c1:
        try:
            selected_passage = bible_df[(bible_df['chapter'] == chapter) & (bible_df['verse']== verse)]
            #st.write(selected_passage)
            passage_details = "{} Chapter::{} Verse::{}".format(book_name,chapter,verse)
            st.info(passage_details)
            passage = "{}".format(selected_passage['text'].values[0])
            st.write(passage)
        except:
            st.warning("Book out of Range")	
        
        with c2:
            #st.success('Verse of the Day')
            chapter_list = range(10)
            verse_list = range(20)
            ch_choice = random.choice(chapter_list)
            vs_choice = random.choice(verse_list)
            random_book_name = random.choice(book_list)
            st.write("Book:{},Ch:{},Vs:{}".format(random_book_name,ch_choice,vs_choice))
            rand_bible_df = df[df["book"] == random_book_name]
            try:
                randomly_selected_passage = rand_bible_df[(rand_bible_df["chapter"] == ch_choice)& (rand_bible_df["verse"] == vs_choice)]
                mytext = randomly_selected_passage["text"].values[0]
            except:
                mytext = rand_bible_df[(rand_bible_df["chapter"] == 1) & (rand_bible_df["verse"] == 1)]["text"].values[0]
                #stc.html(HTML_RANDOM_TEMPLATE.format(mytext), height=300)

        # Search Topic/Term
        search_term = st.text_input("Term/Topic")
        with st.expander("View Results"):
            retrieved_df = df[df["text"].str.contains(search_term)]
            st.dataframe(retrieved_df[["book", "chapter", "verse", "text"]])

elif choice == "MultiVerse":
    st.subheader("MultiVerse Retrieval")
    book_list = df["book"].unique().tolist()
    book_name = st.sidebar.selectbox("Book", book_list)
    chapter = st.sidebar.number_input("Chapter", 1)
    bible_df = df[df["book"] == book_name]
    all_verse = bible_df["verse"].unique().tolist()
    verse = st.sidebar.multiselect("Verse", all_verse, default=1)
    selected_passage = bible_df.iloc[verse]
    #st.dataframe(selected_passage)
    passage_details = "{} Chapter::{} Verse::{}".format(book_name, chapter, verse)
    st.info(passage_details)

    # Layout
    col1, col2 = st.columns(2)
    # Join all text as a sentence
    docx = " ".join(selected_passage["text"].tolist())
    with col1:
        st.info('Details')
        for i,row in selected_passage.iterrows():
            st.write(row['text'])
    
    with col2:
        st.success('Study Mode')
        with st.expander('Visualize Entities'):
            render_entities( docx)
        
        with st.expander("Visualize Pos Tags"):
                tagged_docx = get_tags(docx)
                processed_tags = mytag_visualizer(tagged_docx)
                # st.write(processed_tags)# Raw
                stc.html(processed_tags, height=1000, scrolling=True)
        
        with st.expander("Keywords"):
                processed_docx = nfx.remove_stopwords(docx)
                keywords_tokens = get_most_common_tokens(processed_docx, 5)
                st.write(keywords_tokens)

    with st.expander("Verse Curve"):
        plot_mendelhall_curve(docx)
    
    with st.expander("Word Freq Plot"):
        plot_word_freq_with_altair(docx)
    

    with st.expander("Pos Tags Plot"):
                tagged_docx = get_tags(docx)
                tagged_df = pd.DataFrame(tagged_docx, columns=["Tokens", "Tags"])
                st.dataframe(tagged_df)
                df_tag_count = tagged_df["Tags"].value_counts().to_frame("counts")
                df_tag_count["tag_type"] = df_tag_count.index
                st.dataframe(df_tag_count)

else:
    st.subheader("ABOUT THE APP")
    st.write('''
    A Bible app is a digital platform designed to allow users to access the Bible and
    other faith-based resources from their mobile devices,
    tablets, or computers. The app typically contains the full text of the Bible,
    as well as a range of features that help users engage with the text in new and 
    meaningful ways.

    One of the key benefits of a Bible app is that it provides users with a convenient
    and portable way to access the Bible, no matter where they are.
    Whether they are on the go, traveling, or simply don't have a physical copy of the
    Bible with them, the app allows them to access the text at any time.

    In addition to providing easy access to the Bible, many Bible apps also offer a 
    range of features to help users engage with the text in new and meaningful ways. 

    This can include features such as bookmarking, highlighting, note-taking, and search functionality.
    Overall, a Bible app can be a valuable tool for those looking to deepen their understanding 
    and practice of their faith.
    By providing easy access to the Bible and a range of features to help users engage with the text, 
    the app can help users to grow spiritually and connect with their faith in new and meaningful ways.
    ''')

    #image = Image.open("MY LOGO.jpg")

    #st.image(image,caption=None, width=490, use_column_width=1, clamp=False, channels="RGB", output_format="auto")
    st.subheader("MOTIVE BEHIND THE APP")
    st.write('''
    One wonderful motive for creating the  Bible app is the desire to make
    the Bible more accessible and convenient for people to read and study.
    The Bible is a timeless and powerful source of wisdom and inspiration, but many 
    people find it difficult to access or to carry a physical Bible with them
    at all times.
    
    By creating the Bible app, we can help people overcome these 
    barriers and bring the message of the Bible to more people than ever before.
    
    Finally, creating the  Bible app will be a wonderful way to use our 
    skills and talents to serve others. 
    By combining our knowledge of technology with our passion for the Bible, 
    we  created a tool that helps people to connect with God and experience the 
    transformative power of His word. This can be a deeply rewarding experience, 
    both for us and for the people whose lives are touched by our app.
    ''')
    st.text("A BIBLE APP")
    st.text("POWERED BY:DATA_GEEKS")
    st.image("MY LOGO.jpg")
    st.success("DATA_GEEKS PROJECT")
    st.balloons()
    
   
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


