from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


class MovieSummarizer():
    def __init__(self, model="gpt-4.1-nano-2025-04-14", temperature=0.2, max_tokens=64):
        """
                MovieSummarizer is a wrapper class around langchain_openai.ChatOpenAI, providing the necessary
            system and human/user message templates to make the prediciton. 
        """
        load_dotenv()
        sys_template =( "You are a movie critic who knows every movie's plot. "
                        "You will be requested to summarize a movie provided its name. "
                        "Your summary will be concise, providing only the key premise of the film. "
                        "It will not mention any meta-information, such as cast, crew, the year, etc. "
                        "The summary will start mentioning the film's name and saying what it's about. "
                        "In terms of style, keep a neutral and formal tone, but avoid complicated sentences with many clauses. "
                        "While your summary should avoid vagueness and be detailed, it should also strictly avoid giving away\
                            major plot twists and the ending. "     )
        hum_template = "Summarize the movie {movie_name}."

        # Define the LLM model, along with the prompt (ysing messages)
        self.llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
        self.sys = SystemMessagePromptTemplate.from_template(template=sys_template)
        self.hum = HumanMessagePromptTemplate.from_template(template=hum_template)
        self.prompt = ChatPromptTemplate.from_messages([self.sys, self.hum])
    
    def __call__(self, movie_name: str) -> str:
        """
            Calling an instance of MovieSummarizer class present 
        """
        prompt_val = (self.prompt).invoke({"movie_name":movie_name})
        llm_response = (self.llm).invoke(prompt_val)
        output_raw = llm_response.content
        output_raw = output_raw.split('.')
        output_final = ".".join(output_raw) if output_raw[-1] == "" else ".".join(output_raw[:-1])
        if len(output_final) > 0:
            return output_final
        else:
            return "An error occured. Please try again"
        

if __name__ == "__main__":
    movie_summ = MovieSummarizer()
    movie_name = "Mulholland Drive"
    summary = movie_summ(movie_name)
    print(summary)



