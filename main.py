import yaml
import ollama
import argparse
import bs4
from pydantic import BaseModel
from typing import Dict, Optional, List
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma


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
        documents = []
        for pages in self.configuration["DataSources"]["wiki"]:
            print("Loading Wiki Page: ", pages)
            wiki = WikipediaLoader(query=pages, load_max_docs=10)
            wiki_content = wiki.load()
            documents.append(wiki_content)
        return documents

    def load_html(self) -> list:
        """This functions loads user data from html pages"""
        documents = []
        for pages in self.configuration["DataSources"]["html"]:
            print("Loading HTML Page: ", pages)
            html = BSHTMLLoader(file_path=pages, open_encoding="utf8")
            html_content = html.load()
            documents.append(html_content)
        return documents

    def load_webpage(self) -> List:
        """This functions loads user data from webpages"""
        documents = []
        for pages in self.configuration["DataSources"]["webpage"]:
            print("Loading WebPage: ", pages)
            loader = WebBaseLoader(pages)
            data = loader.load()
            documents.append(data)
        return documents

    def load_database(self) -> None:
        """this function will load all data to chromadb"""
        Documents = []
        Documents += my_llm.load_wiki()
        Documents += my_llm.load_html()
        Documents += my_llm.load_webpage()
        print("Total number of documents loaded: ", len(Documents))
        text_splitter = RecursiveCharacterTextSplitter()
        ollama_emb = OllamaEmbeddings(model=self.configuration["LLM"])
        vectorstore = Chroma("LLM_Store", ollama_emb)
        for doc in Documents:
            splitted_docs = text_splitter.split_documents(doc)
            vectorstore.add_documents(doc)
        vectorstore.persist()
        print("Data successfully loaded to chromaDB")

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
    my_llm.load_database()
    # my_llm.rag_chain("What is Gita?")
