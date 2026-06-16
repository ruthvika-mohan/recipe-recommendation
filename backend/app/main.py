from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from .feedback import FeedbackStore, build_feedback_store
from .recommender import RecipeRecommender
from .schemas import (
    FeedbackRequest,
    FeedbackResponse,
    MetadataResponse,
    OrderIntentRequest,
    OrderIntentResponse,
    ProfileResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecipeDetail,
    UserProfile,
)
from .settings import Settings, get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.recommender = RecipeRecommender(settings.project_dir)
    app.state.feedback_store = build_feedback_store(settings)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata", response_model=MetadataResponse)
    async def metadata(recommender: RecipeRecommender = Depends(get_recommender)) -> MetadataResponse:
        return recommender.metadata()

    @app.post("/recommendations", response_model=RecommendationResponse)
    async def recommendations(
        payload: RecommendationRequest,
        recommender: RecipeRecommender = Depends(get_recommender),
        feedback_store: FeedbackStore = Depends(get_feedback_store),
    ) -> RecommendationResponse:
        return await recommender.recommend(payload, feedback_store)

    @app.get("/recipes/{recipe_id}", response_model=RecipeDetail)
    async def recipe_detail(
        recipe_id: int, recommender: RecipeRecommender = Depends(get_recommender)
    ) -> RecipeDetail:
        try:
            return recommender.recipe_detail(recipe_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Recipe not found") from exc

    @app.post("/feedback", response_model=FeedbackResponse)
    async def feedback(
        payload: FeedbackRequest,
        authorization: str | None = Header(default=None),
        x_user_id: str | None = Header(default=None),
        feedback_store: FeedbackStore = Depends(get_feedback_store),
    ) -> FeedbackResponse:
        user_id = payload.user_id or x_user_id or user_id_from_authorization(authorization)
        if not user_id:
            raise HTTPException(status_code=401, detail="Feedback requires an authenticated user")
        stored = await feedback_store.store_feedback(user_id, payload)
        return FeedbackResponse(stored=stored, provider=feedback_store.provider)

    @app.get("/profile/{user_id}", response_model=ProfileResponse)
    async def profile(
        user_id: str,
        feedback_store: FeedbackStore = Depends(get_feedback_store),
    ) -> ProfileResponse:
        profile_data = await feedback_store.get_profile(user_id)
        return ProfileResponse(profile=profile_data, provider=feedback_store.provider)

    @app.put("/profile/{user_id}", response_model=ProfileResponse)
    async def update_profile(
        user_id: str,
        payload: UserProfile,
        feedback_store: FeedbackStore = Depends(get_feedback_store),
    ) -> ProfileResponse:
        if payload.user_id != user_id:
            raise HTTPException(status_code=400, detail="Profile user_id must match URL user_id")
        profile_data = await feedback_store.upsert_profile(payload)
        return ProfileResponse(profile=profile_data, provider=feedback_store.provider)

    @app.post("/order-intent", response_model=OrderIntentResponse)
    async def order_intent(payload: OrderIntentRequest) -> OrderIntentResponse:
        items = sorted({item.strip().lower() for item in payload.missing_ingredients if item.strip()})
        provider = "blinkit" if payload.provider in {"any", "blinkit"} else "swiggy"
        query = "+".join([provider, *items])
        search_url = f"https://www.google.com/search?q={query}" if items else None
        return OrderIntentResponse(
            provider=provider,
            status="integration_required",
            message=(
                "Order API credentials are not configured yet. Use this missing-items list to build a "
                "Blinkit/Swiggy Instamart cart once partner API access is available."
            ),
            items=items,
            search_url=search_url,
        )

    return app


def get_recommender(request: Request) -> RecipeRecommender:
    return request.app.state.recommender


def get_feedback_store(request: Request) -> FeedbackStore:
    return request.app.state.feedback_store


def user_id_from_authorization(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return f"bearer:{token[-16:]}"


app = create_app()
