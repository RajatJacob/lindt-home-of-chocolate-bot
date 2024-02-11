from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from datetime import datetime
from agent import Agent
from entity import Entity
from retriever import search_and_vectorize
import logging


class Role(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'


class Message(BaseModel):
    sender: Role
    text: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Conversation(BaseModel):
    conversation_id: int
    history: List[Message]
    agent: Agent

    @classmethod
    def start(cls, agent: Agent, history: List[Message] | None = None):
        conversation = cls(
            conversation_id=int(datetime.now().timestamp()),
            history=list(history or []),
            agent=agent
        )
        return conversation

    def find_context(self, question: str, max_char=512) -> str:
        entities = Entity.from_text(question)
        history = '\n'.join([message.text for message in self.history])[
            :max_char//2]
        out = []
        for entity in entities:
            logging.info(f'Searching for "{entity.name}"')
            docs = search_and_vectorize(entity.name)
            for doc in docs['documents']:
                out.extend(doc)
        return history + ('\n\n'.join(out)[max_char//2:])

    def send(self, text: str):
        message = Message(sender=Role.USER, text=text)
        self.history.append(message)
        context = self.find_context(text)
        self.history.append(Message(sender=Role.SYSTEM, text=context))
        response = self.agent.ask(text, context)
        response_message = Message(sender=Role.ASSISTANT, text=response)
        self.history.append(response_message)
        return response_message