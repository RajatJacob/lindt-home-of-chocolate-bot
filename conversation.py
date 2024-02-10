from pydantic import BaseModel, Field
from typing import List
from enum import Enum
from datetime import datetime
from agent import Agent


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

    def send(self, text: str):
        message = Message(sender=Role.USER, text=text)
        self.history.append(message)
        response = self.agent.ask(text)
        response_message = Message(sender=Role.ASSISTANT, text=response)
        self.history.append(response_message)
        return response_message
