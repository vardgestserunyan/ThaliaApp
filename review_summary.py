from bs4 import BeautifulSoup, SoupStrainer
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import requests


class ReviewFetcher():
    def __init__(self, search_name):
        self.search_name = search_name

    def find_and_review(self):
        for movie_title, review_url in self.find_movie():
            if movie_title == "NONE FOUND":
                return ("No movies found with that search term.", None)
            if movie_title == "ENDLIST":
                return ("No more movies found with that search term.", None)
            return (movie_title, self.get_reviews(review_url))
        
    
    def find_movie(self):    
        movie_name = (self.search_name).replace(" ", "%20")
        url = f"https://www.rottentomatoes.com/search?search={movie_name}"
        strainer = SoupStrainer("search-page-media-row")
        raw_html = (requests.get(url)).content
        soup = BeautifulSoup(raw_html, "html.parser", parse_only=strainer)
        search_results = soup.find_all(attrs={"slot":"title"})
        
        if search_results == []:
            return ("NONE FOUND", None)

        for result in search_results:
            movie_title = result.get_text(strip=True)
            reviews_url = result.attrs["href"]+"/reviews"
            yield movie_title, reviews_url
        
        return ("ENDLIST", None)

    def get_reviews(self, url):
        raw_html = (requests.get(url)).content
        strainer = SoupStrainer(class_="review-row")
        soup = BeautifulSoup(raw_html, "html.parser", parse_only=strainer)

        negative_list, positive_list = [], []
        for review_obj in soup:
            review_text_obj = review_obj.find(class_="review-text")
            review_text = review_text_obj.get_text()
            positive = True if review_obj.find(attrs={"sentiment":"POSITIVE"}) else False
            if positive:
                positive_list.append(review_text)
            else:
                negative_list.append(review_text)
        
        return positive_list, negative_list


class ReviewSummarizer():
    def __init__(self, model="gpt-4.1-nano-2025-04-14", temperature=0.2, max_tokens=64):
        load_dotenv()
        msg = [ SystemMessagePromptTemplate.from_template(template=("You will receive a set of {sentiment} movie reviews."
                                                                    "You will summarize those reviews."
                                                                    "You will be brief, but will emphasize all the specific adjectives and descriptors."
                                                                    "You will pay special attention to aspects that recur across reviews."
                                                                    "Your response will emphasize the {sentiment} aspect in he rviews.")),
                SystemMessagePromptTemplate.from_template(template="Here you will find the list of {sentiment} reviews: {list}"),
                HumanMessagePromptTemplate.from_template(template="Please summarize the reviews based on the criteria you received.")

        ]
        self.llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
        self.prompt = ChatPromptTemplate.from_messages(msg)

    def __call__(self, sentiment, review_list):
        if len(review_list) == 0:
            return f"No {sentiment} reviews were found."
        
        prompt_val = (self.prompt).invoke({"sentiment":sentiment,"list":review_list})
        response = (self.llm).invoke(prompt_val)
        output_raw = (response.content).split(".")
        output = ". ".join(output_raw) if output_raw[-1] == "" else ". ".join(output_raw[:-1])
        if len(output)>0:
            return output
        else:
            return "An error occured. Please try again"




search_name = "oierhoihgo"
review_fetch_obj = ReviewFetcher(search_name)
movie_title, (positive_list, negative_list) = review_fetch_obj.find_and_review()


review_smmrz_obj = ReviewSummarizer()
positive_summ = review_smmrz_obj("positive", positive_list)
negative_summ = review_smmrz_obj("negative", negative_list)
