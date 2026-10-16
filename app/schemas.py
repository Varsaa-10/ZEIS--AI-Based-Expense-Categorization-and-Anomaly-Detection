"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from src import config as C


class CategorizeRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, json_schema_extra={"example": {"description": "flight to Mumbai", "merchant": "IndiGo"}})
    description: str = Field(..., max_length=500)
    merchant: Optional[str] = Field(default="", max_length=200)


class TopCategory(BaseModel):
    category: str
    score: float


class CategorizeResponse(BaseModel):
    predicted_category: str
    category_score: float = Field(..., description="Model confidence score in [0,1]; NOT a calibrated probability")
    top_3: list[TopCategory]
    model_version: str


class ExpenseRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, json_schema_extra={"example": {
        "employee_id": "E0042", "department": "Sales", "expense_date": "2025-01-10", "merchant": "Swiggy",
        "description": "team lunch", "category": "Meals", "amount": 620.0, "payment_method": "UPI"}})
    employee_id: str = Field(..., min_length=1, max_length=64)
    department: Optional[str] = Field(default=None, max_length=100)
    expense_date: date
    merchant: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=500)
    category: Optional[str] = Field(default=None, max_length=100, description="Category claimed by the employee (optional)")
    amount: float = Field(..., gt=0, le=C.MAX_AMOUNT, allow_inf_nan=False)
    payment_method: Optional[str] = Field(default=None, max_length=100)


class ReasonCode(BaseModel):
    code: str
    severity: Literal["signal", "info"]
    message: str


class AnalyzeResponse(BaseModel):
    predicted_category: str
    category_score: float
    claimed_category: Optional[str]
    anomaly_score: float = Field(..., description="Model score in [0,1]; NOT a calibrated probability")
    risk_tier: Literal["approve", "review", "flag"]
    reason_codes: list[ReasonCode]
    model_version: str


class BatchRequest(BaseModel):
    expenses: list[ExpenseRequest] = Field(..., min_length=1, max_length=C.MAX_BATCH)


class BatchResponse(BaseModel):
    count: int
    summary: dict[str, int]
    results: list[AnalyzeResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    model_version: Optional[str] = None
    model_a: Optional[str] = None
    model_b: Optional[str] = None
