from pydantic import BaseModel


class WarningItem(BaseModel):
    code: str
    message: str
