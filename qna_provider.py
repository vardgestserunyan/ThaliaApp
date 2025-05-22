from bs4 import BeautifulSoup, SoupStrainer
from langchain_text_splitters import HTMLSectionSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import requests

class RetrieverIMDB:
    def __init__(self, search_name):
        self.search_name = search_name.replace(" ", "%20")
        self.search_titles = self.imdb_searcher()

    def imdb_searcher(self):
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
        search_url = f"https://www.imdb.com/find/?q={self.search_name}"
        raw_html = (requests.get(search_url, headers=headers)).content
        strainer = SoupStrainer(class_="sc-31dae308-2 jtgVzb")
        soup = BeautifulSoup(raw_html, "html.parser", parse_only=strainer)
        search_titles = soup.find_all(class_="ipc-metadata-list-summary-item__t")
        
        return search_titles

    def url_retriever(self, order=0):
        if len(self.search_titles) == 0:
            return "No movies found with that name."
        if order >= len(self.search_titles):
            return "End of search results."
        search_item = self.search_titles[order]
        movie_page = (search_item.attrs).get("href")
        movie_url = f"https://www.imdb.com{movie_page}"

        return movie_url

class QuestionAnswerer:
    def __init__(self, movie_name, embed_model="text-embedding-3-small", chat_model="gpt-4.1-nano-2025-04-14"):
        self.movie_name = movie_name
        self.embed_model = OpenAIEmbeddings(model=embed_model, dimensions=128)
        self.chat_model = ChatOpenAI(model=chat_model, temperature=0.2, max_tokens=64)
        self.splitter = HTMLSectionSplitter(headers_to_split_on=[("section","section"), ("li","list")], 
                                            max_chunk_size=1000, chunk_overlap=100)
        self.vector_store = self.knowledge_extractor()
        self.sys_msgs = [SystemMessagePromptTemplate.from_template(("You will answer a question about a movie. "
                                                                    "For help, you will receive the name of the movie "
                                                                        "and relevant contextual information in two pieces."
                                                                    "Your answer will be concise and to the point. "
                                                                    "If you cannot find the information in your knowledge base "
                                                                        "or in the context, you will say 'I don't know'. "
                                                                    "If the question is not relevant to the movie, " 
                                                                        "say 'I don't think that's relevant'. You will say that "
                                                                        "only if you cannot find anything on the query in the context.")),
                         SystemMessagePromptTemplate.from_template("The name of the movie is {movie_name}"),
                         SystemMessagePromptTemplate.from_template("This is contextual information, Chunk 1: \n {context_chunk_1} "),
                         SystemMessagePromptTemplate.from_template("This is contextual information, Chunk 2: \n {context_chunk_2} ") ]

    def knowledge_extractor(self):
        headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win 64 ; x64) Apple WeKit /537.36(KHTML , like Gecko) Chrome/80.0.3987.162 Safari/537.36'}
        sections = ["UserReviews", "awards", "title-cast", "title-cast-item", "title-details-header",
                    "MoreLikeThis", "DidYouKnow", "faq-content", "Details", "BoxOffice","TechSpecs"]
        strainer = SoupStrainer(attrs={"data-testid":sections})
        raw_html = (requests.get(movie_url,headers=headers)).content
        webpage_soup = BeautifulSoup(raw_html, "html.parser", parse_only=strainer)
        knowledge_base = webpage_soup.find_all(["section", "li"],attrs={"data-testid":sections})
        knowledge_base = [html_chunk.prettify() for html_chunk in knowledge_base]
        knowledge_base = "\n".join(knowledge_base)
        knowledge_base = (self.splitter).split_text(knowledge_base)
        vector_store = InMemoryVectorStore(self.embed_model)
        vector_store.add_documents(knowledge_base)

        return vector_store


    def __call__(self, question):
        retrieved_info = (self.vector_store).similarity_search(query=question, k=2)
        context_chunk_1, context_chunk_2 = retrieved_info[0], retrieved_info[1]
        full_msg = self.sys_msgs + [HumanMessagePromptTemplate.from_template("{question}")]
        prompt = ChatPromptTemplate.from_messages(full_msg)
        prompt_val = prompt.invoke({"movie_name": self.movie_name, "context_chunk_1": context_chunk_1, 
                                    "context_chunk_2": context_chunk_2, "question": question})
        llm_response = (self.chat_model).invoke(prompt_val)

        return llm_response.content




if __name__ == "__main__":
    search_name, order = "Mulholland Drive", 5
    retriever = RetrieverIMDB(search_name)
    movie_url = retriever.url_retriever(order)

    if movie_url == "End of search results." or movie_url == "No movies found with that name.":
        print(movie_url)
    
    else:
        question = "What's the name of Naomi Watt's Character?"
        QnA = QuestionAnswerer(search_name)
        answer = QnA(question)
        print(answer)

