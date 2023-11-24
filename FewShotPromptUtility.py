import os
from langchain import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai


class FewShotPromptUtility:

    def __init__(self, examples, prefix, suffix, input_variables, example_template, example_variables):
        self.examples = examples
        self.prefix = prefix
        self.suffix = suffix
        self.input_variables = input_variables
        self.example_template = example_template
        self.example_variables = example_variables

    def get_prompt_template(self):
        example_prompt = PromptTemplate(
            input_variables=self.example_variables,
            template=self.example_template
        )
        return example_prompt
    
    def get_embeddings(self):
        return embeddings = OpenAIEmbeddings(openai_api_key="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")
    
    def get_example_selector(self):
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            get_prompt_template(),
            get_embeddings(),
            FAISS,
            k=3
        )
        
    def get_prompt(self, question):
        prompt_template = FewShotPromptTemplate(
            example_selector=get_example_selector(),
            example_prompt=self.get_prompt_template(),
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_variables
        )
        prompt = prompt_template.format(question=question)
        return prompt