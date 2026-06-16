from typing import Literal

from pydantic import BaseModel, Field


FeedbackAction = Literal["like", "dislike", "save", "cooked", "not_relevant"]


class RecommendationRequest(BaseModel):
    cuisine: str | None = None
    diet: str | None = None
    course: str | None = None
    max_total_time_mins: int | None = Field(default=None, ge=1)
    ingredients: list[str] = Field(default_factory=list)
    pantry_ingredients: list[str] = Field(default_factory=list)
    fridge_ingredients: list[str] = Field(default_factory=list)
    unavailable_ingredients: list[str] = Field(default_factory=list)
    liked_recipe_ids: list[int] = Field(default_factory=list)
    surprise: bool = False
    user_id: str | None = None
    limit: int = Field(default=12, ge=1, le=50)


class ScoreBreakdown(BaseModel):
    final_score: float
    match_label: str
    match_reason: str
    ingredient_coverage: float
    recipe_coverage: float
    tfidf_similarity: float
    cuisine_match: float
    course_match: float
    feedback_boost: float
    matched_ingredients: list[str]
    missing_ingredients: list[str]
    alternate_course: bool
    alternate_cuisine: bool


class RecipeSummary(BaseModel):
    recipe_id: int
    name: str
    cuisine: str
    course: str
    diet: str
    prep_time_mins: int | None = None
    cook_time_mins: int | None = None
    total_time_mins: int | None = None
    hidden_prep_time_mins: int | None = None
    effective_total_time_mins: int | None = None
    time_note: str | None = None
    servings: int | None = None
    ingredients: list[str]
    url: str | None = None
    score: ScoreBreakdown


class RecommendationResponse(BaseModel):
    results: list[RecipeSummary]
    total_candidates: int
    strict_diet_applied: bool


class RecipeDetail(BaseModel):
    recipe_id: int
    name: str
    cuisine: str
    course: str
    diet: str
    prep_time_mins: int | None = None
    cook_time_mins: int | None = None
    total_time_mins: int | None = None
    hidden_prep_time_mins: int | None = None
    effective_total_time_mins: int | None = None
    time_note: str | None = None
    servings: int | None = None
    ingredients: list[str]
    instructions: list[str]
    url: str | None = None


class MetadataResponse(BaseModel):
    cuisines: list[str]
    diets: list[str]
    courses: list[str]
    ingredients: list[str]


class FeedbackRequest(BaseModel):
    recipe_id: int
    action: FeedbackAction
    user_id: str | None = None
    reason: str | None = None
    comment: str | None = None
    context: dict = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    stored: bool
    provider: str


class UserProfile(BaseModel):
    user_id: str
    dietary_preference: str | None = None
    pantry_ingredients: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)


class ProfileResponse(BaseModel):
    profile: UserProfile
    provider: str


class OrderIntentRequest(BaseModel):
    user_id: str | None = None
    recipe_id: int | None = None
    provider: Literal["blinkit", "swiggy", "any"] = "any"
    missing_ingredients: list[str] = Field(default_factory=list)
    delivery_location: str | None = None


class OrderIntentResponse(BaseModel):
    provider: str
    status: Literal["ready", "integration_required"]
    message: str
    items: list[str]
    search_url: str | None = None
