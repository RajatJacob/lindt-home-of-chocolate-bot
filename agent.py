import os
from dotenv import load_dotenv
from abc import ABC
from enum import Enum
from typing import Type
from transformers import pipeline
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("Environment variable `GEMINI_API_KEY` is not set")

genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel('gemini-pro')


flan = pipeline("text2text-generation", model="google/flan-t5-large")
question_answering = pipeline(
    'question-answering',
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2"
)


class _AgentABC(ABC):
    @classmethod
    def ask(cls, question: str, context: str) -> str:
        raise NotImplementedError()

    @classmethod
    def _question_context_to_prompt(
            cls,
            question: str,
            context: str | None = None
    ) -> str:
        if not context:
            return question
        return f"""Context: {context}\nQuestion: {question}"""


class QAAgent(_AgentABC):
    @classmethod
    def ask(cls, question: str, context: str):
        return question_answering(
            question=question,
            context=context
        )['answer']


class FlanAgent(_AgentABC):
    @classmethod
    def ask(cls, question: str, context: str | None = None):
        prompt = cls._question_context_to_prompt(question, context)
        return flan(prompt, max_new_tokens=1000)[0]['generated_text']


class GeminiAgent(_AgentABC):
    @classmethod
    def ask(cls, question: str, context: str | None = None):
        prompt = cls._question_context_to_prompt(question, context)
        response = gemini_model.generate_content(prompt)
        return response.text


class Agent(Enum):
    FLAN = 'flan'
    GEMINI = 'gemini'
    QA = 'qa'

    def get_agent(self) -> Type[_AgentABC]:
        return {
            Agent.FLAN: FlanAgent,
            Agent.GEMINI: GeminiAgent,
            Agent.QA: QAAgent
        }[self]

    def ask(self, question: str, context: str) -> str:
        return str(self.get_agent().ask(question, context))
