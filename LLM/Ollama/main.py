import yaml
import ollama
import argparse
from pydantic import BaseModel
from typing import Dict, Optional, List
from langchain_community.document_loaders.wikipedia import WikipediaLoader


class ConfigureableLLM(BaseModel):

    config_path: str
    configuration: Optional[Dict] = None

    def load_config(self) -> None:
        """This functions loads the config file"""
        with open(self.config_path, "r") as file:
            self.configuration = yaml.safe_load(file)

    def load_llm(self, question: str, context: str) -> list:
        """This functions enables and loads ollama"""
        print("Selected LLM Model is: ", self.configuration["LLM"])
        if self.configuration["RAG"] == "Enable":
            query = f"context: {context}/n/n question: {question}"
        else:
            query = f"question: {question}"
        response = ollama.chat(
            model=self.configuration["LLM"],
            messages=[
                {"role": "user", "content": query},
            ],
        )
        return response["message"]["content"]

    def load_wiki(self) -> list:
        """This function loads user data provided in different formats"""
        for pages in self.configuration["DataSources"]["wiki"]:
            print("Loading Wiki Page: ", wiki)
            wiki = WikipediaLoader(query=pages, load_max_docs=10)
            wiki.load()
            print(wiki)

    def rag_chain(self, question: str) -> str:
        if self.configuration["RAG"] == "Enable":
            print("RAG is Enabled.")
        else:
            print("RAG is disabled.")
            self.load_llm(question, "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Pass configuration file"
    )
    args = parser.parse_args()
    my_llm = ConfigureableLLM(**{"config_path": args.config})
    my_llm.load_config()
    my_llm.load_wiki()
    # my_llm.rag_chain("What is Gita?")
