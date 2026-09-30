"""Pydantic input schemas."""
from typing import Optional

from pydantic import BaseModel, Field


class RegisterUser(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, pattern=r"^[A-Za-z0-9_]+$")
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    full_name: Optional[str] = Field(None, max_length=80)
    password: str = Field(..., min_length=6, max_length=72)


class HomeBudgetInput(BaseModel):
    total_budget: float = Field(..., gt=0, le=1_000_000_000)
    num_lights: int = Field(0, ge=0, le=100)
    num_fans: int = Field(0, ge=0, le=50)
    num_furniture: int = Field(0, ge=0, le=100)
    num_dining_tables: int = Field(0, ge=0, le=20)
    has_living_room: bool = False
    has_kitchen: bool = False
    has_bedroom: bool = False
    additional_requirements: Optional[str] = Field(None, max_length=1000)


class PartyBudgetInput(BaseModel):
    total_budget: float = Field(..., gt=0, le=1_000_000_000)
    num_guests: int = Field(..., ge=1, le=5000)
    party_type: str = Field("Birthday", min_length=1, max_length=50)
    venue_type: Optional[str] = Field("Home", max_length=50)
    needs_catering: bool = True
    needs_decoration: bool = True
    needs_entertainment: bool = True
    additional_requirements: Optional[str] = Field(None, max_length=1000)


class JewelryBudgetInput(BaseModel):
    total_budget: float = Field(..., gt=0, le=1_000_000_000)
    occasion: str = Field(..., min_length=1, max_length=80)
    preferences: Optional[str] = Field(None, max_length=1000)