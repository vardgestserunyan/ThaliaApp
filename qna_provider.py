from bs4 import BeautifulSoup, SoupStrainer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
import requests


search_name, order, question = "Mulholland Drive", 0, "Who's the writer?"
search_name = search_name.replace(" ", "%20")
headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
search_url = f"https://www.imdb.com/find/?q={search_name}"
raw_html = (requests.get(search_url, headers=headers)).content
strainer = SoupStrainer(class_="sc-31dae308-2 jtgVzb")
soup = BeautifulSoup(raw_html, "html.parser", parse_only=strainer)
search_titles = soup.find_all(class_="ipc-metadata-list-summary-item__t")
search_item = search_titles[order]
movie_name = search_item.get_text()
movie_page = (search_item.attrs).get("href")
movie_url = f"https://www.imdb.com{movie_page}"


info_sections = ["atf--bg", "awards","title-cast", 
                 "UserReviews", "MoreLikeThis","DidYouKnow",
                 "faq-content","Details","BoxOffice", "TechSpecs","News"]
strainer = SoupStrainer(role="main")

raw_html = (requests.get(movie_url, headers=headers)).content
parsed_soup = BeautifulSoup(raw_html, "html.parser", parse_only=strainer)
parsed_info_list = [ parsed_soup.find(attrs={"data-testid":split_id}) for split_id in info_sections ]
parsed_info_list = [ parsed_tag.get_text(separator="\t") for parsed_tag in parsed_info_list if parsed_tag]
info_base = "\n\n".join(parsed_info_list)

splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=500, chunk_overlap=10)
split_text = splitter.split_text(info_base)


llm_emb = OpenAIEmbeddings(model="text-embedding-3-small",
                           dimensions=64)
vector_store = InMemoryVectorStore(llm_emb)
vector_store.add_texts(split_text)
context = (vector_store.similarity_search(question,k=1))[0]


msgs = [ SystemMessagePromptTemplate.from_template(("Answer the following question based on the context information retrieved."
                                                    "In doing this, be mindful of the fact that the information is parsed from IMDB."
                                                    "As such, it might not be ccleanly parsed, so base the final answer on the document..."
                                                    "... as well as your prior knowledge."
                                                    "Be concise and informative, and if you aren't sure, just say 'I don't know'.")),
          SystemMessagePromptTemplate.from_template("Here is the context from IMDB: {context}"),
          HumanMessagePromptTemplate.from_template("{question}")  ]

prompt = ChatPromptTemplate.from_messages(msgs)
prompt_val = prompt.invoke({"context":context, "question":question})

llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.2, max_tokens=64)
result = (llm.invoke(prompt_val)).content





# Got the info base -- now proceeed to making the vector DB and making a RAG
# No need to parse the webpage -- WebBaseLoader does it already!
# join method
info_base_list[0].contents