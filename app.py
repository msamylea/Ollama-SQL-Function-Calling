from CustomOllamaFunctions import CustomOllamaFunctions
import sql_handler

llm = "l3custom"

system_prompt = """You are a perfectionist who is logical, follows directions strictky, and is very professional.
            You job is to help everyone as much as possible.

            You have access to the following tools:

            {tools}

            You must always select a tool. Do not add additional text or include "Response" or respond in strings.
            Your response should be in the following JSON format with no other included text or data:

            {{
            "tool": <name of the selected tool>,
            "tool_input": <parameters for the selected tool, matching the tool's JSON schema. Ensure the format matches correctly.>
            }}

            """

chat_llm: CustomOllamaFunctions = CustomOllamaFunctions(
    functions=[
        sql_handler.SQLFunction(model=llm)
        ], 
    model=llm,
    prompt_template=system_prompt)

chat_llm.run(input="Use the SQL database to return the number of customers per country.")
