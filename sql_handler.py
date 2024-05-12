from Function import BaseFunction, Parameter, ParameterType, Property, PropertyType
from langchain_community.llms import Ollama
import sqlite3
import re

class SQLFunction(BaseFunction):
    """
    A function derived from the BaseFunction class that allows the LLM to interact with a SQLite database.
    """
  
    def __init__(self, model: str):
        """
        Instantiates a new SQLFunction object with the given model.

        Args:
            model (str): The name of the Ollama model to use.
        """
        self.llm: Ollama = Ollama(model="l3custom")
        super().__init__(name="query_schema",
                         description="Get the schema for the SQL database to use in query generation.",
                         parameters=[
                             Parameter(type=ParameterType.OBJECT,
                                       properties=[
                                           Property(name="input",
                                                    type=PropertyType.STRING,
                                                    attribute={'description': 'The user input to generate a SQL query after getting the schema.'}),
                                       ],
                                       required_parameters=["input"])
                         ])

    db = 'chinook.db'

    def run_query(self, query):
        with sqlite3.connect(self.db) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            print("Result:", result)
            return result
    
    def get_schema(self)->list:
        conn = sqlite3.connect(self.db)
        sql_query = """SELECT name FROM sqlite_master  
                        WHERE type='table';"""
       
        cursor = conn.cursor()
        cursor.execute(sql_query)
        schema = cursor.fetchall()
        return schema

    def __call__(self, arguments: dict) -> str:
        """
        Queries the SQLite database using the given input and returns the result.

        Args:
            arguments (dict): The arguments for the function, containing the user input.

        Returns:
            str: The result of the SQL query.
        """
        
        schema = self.get_schema()
        print("Schema:", schema)
        schema_str = ', '.join([str(elem) for elem in schema])  

        user_input = arguments["input"]
        print("User Input:", user_input)
        llm_response = self.llm(f"Using ONLY the table schema provided, write a SQL query to answer the user's request. You MUST ONLY use the tables in the schema. Do not make up tables or abbreviate or change tables:\nTables: {schema_str}\nUser Input: {user_input}\nSQL Query:")
        print("LLM Response:", llm_response)
        match = re.search(r"(SELECT.*;)", llm_response)
        if match:
            sql_query = match.group(1)
        if "```sql" in llm_response:
            sql_query = llm_response.split("```sql")[1].split("```")[0].strip()

        result = self.run_query(sql_query)

        return f"The result of the SQL query is: {result}"