from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_community.embeddings import OllamaEmbeddings 
from chromadb.config import Settings
from chromadb import Client
from langchain.memory import VectorStoreRetrieverMemory  
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.messages.base import BaseMessage
from Function import BaseFunction, functions_to_json

class CustomOllamaFunctions:
    """
    A class representing a ChatOllamaFunctions object.

    Attributes:
        model (OllamaFunctions): The Ollama functional model used by the ChatOllamaFunctions object.
        functions (list[BaseFunction]): List of functions/tools the ChatOllamaFunctions object can use.

    Methods:
        __init__(self, functions: list[BaseFunction], model: str, prompt_template: str = None):
            Initializes a ChatOllamaFunctions object.
        __load_functions(self, _functions: list[BaseFunction]) -> None:
            Binds all functions to the Ollama functional model.
        __get_vector_memory(self, model:str, chroma_dir: str = "chroma_db", collection_name: str = "default") -> VectorStoreRetrieverMemory:
            Connects to a ChromaDB vector store database and instantiates a VectorStoreRetrieverMemory object.
        invoke(self, input: str) -> BaseMessage:
            Invokes the Ollama functional model with the given input.
        run(self, input: str) -> str:
            Runs the Ollama functional model with the given input and returns the response from the invoked function.
    """

    model: OllamaFunctions
    functions: list[BaseFunction]
    
    def __init__(self, functions: list[BaseFunction], model: str, prompt_template: str = None):
        """
        Initializes a ChatOllamaFunctions object.

        Args:
            functions (list[BaseFunction]): List of functions/tools the ChatOllamaFunctions object can use.
            model (str): The name of the Ollama functional model to use.
            prompt_template (str, optional): Optional prompt the ChatOllamaFunctions object can use for picking 
                which function to use. Defaults to None.
        """
        self.model = OllamaFunctions(model=model, tool_system_prompt_template=prompt_template)
        self.functions = []
        self.__load_functions(_functions=functions)
        
    def __load_functions(self, _functions: list[BaseFunction]) -> None:
        """
        __load_functions is a private function that binds all functions to the llm.

        Args:
            _functions (list[BaseFunction]): List of functions/tools the llm can use.
        """
        self.functions = _functions
        self.model = self.model.bind(functions=functions_to_json(funcs=self.functions))
    
    def __get_vector_memory(self, model:str, chroma_dir: str = "chroma_db", collection_name: str = "default") -> VectorStoreRetrieverMemory:
        """
        __get_vector_memory is a private function that connects to a ChromaDB vector store database and instantiates VectorStoreRetrieverMemory object.

        Args:
            model (str): The name of the OllamaEmbeddings model to use.
            chroma_dir (str, optional): The directory where ChromaDB data will be stored. Defaults to "chroma_db".
            collection_name (str, optional): The name of the collection to use in the database. Defaults to "default".

        Returns:
            VectorStoreRetrieverMemory: The instantiated VectorStoreRetrieverMemory object.
        """
        embeddings: OllamaEmbeddings = OllamaEmbeddings(model=model)

        # Configure ChromaDB settings
        chroma_config = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_dir,
            anonymized_telemetry=False
        )

        # Initialize the ChromaDB client
        chroma_client = Client(chroma_config)

        # Get the collection or create a new one if it doesn't exist
        chroma_collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embeddings.embed_query
        )

        # Define the retriever for the vector store.
        retriever: VectorStoreRetriever = chroma_collection.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 50})

        # Define the memory for the vector store.
        vector_memory: VectorStoreRetrieverMemory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="history", input_key="input")

        return vector_memory
    
    def invoke(self, input: str) -> BaseMessage:
        """
        Invokes the Ollama functional model with the given input.

        Args:
            input (str): The input to invoke the Ollama functional model with.

        Returns:
            BaseMessage: The response from the Ollama functional model.
        """
        return self.model.invoke(input)
    
    def run(self, input: str) -> str:
        resp: BaseMessage = self.invoke(input)
        resp_function_call: dict = resp.additional_kwargs["function_call"]
        function_name: str = resp_function_call["name"]
        function_arguments: dict = {"input": input}
        for function in self.functions:
            if function.name == function_name:
                print("function:", function)
                print("function_arguments:", function_arguments)
                return function(function_arguments)
        
        return "Error: No matching function found for the given input."