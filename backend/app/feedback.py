from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

import httpx

from .schemas import FeedbackRequest, UserProfile
from .settings import Settings


POSITIVE_ACTIONS = {"like", "save", "cooked"}
NEGATIVE_ACTIONS = {"dislike", "not_relevant"}


class FeedbackStore:
    provider = "memory"

    async def get_boosts(self, user_id: str | None, recipe_ids: list[int]) -> dict[int, float]:
        return {recipe_id: 0.0 for recipe_id in recipe_ids}

    async def store_feedback(self, user_id: str, feedback: FeedbackRequest) -> bool:
        return True

    async def store_recommendation_event(
        self, user_id: str | None, request_payload: dict[str, Any], recipe_ids: list[int]
    ) -> None:
        return None

    async def get_profile(self, user_id: str) -> UserProfile:
        return UserProfile(user_id=user_id)

    async def upsert_profile(self, profile: UserProfile) -> UserProfile:
        return profile


class MemoryFeedbackStore(FeedbackStore):
    provider = "memory"

    def __init__(self) -> None:
        self._global_counts: dict[int, Counter[str]] = defaultdict(Counter)
        self._user_counts: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        self._profiles: dict[str, UserProfile] = {}

    async def get_boosts(self, user_id: str | None, recipe_ids: list[int]) -> dict[int, float]:
        boosts: dict[int, float] = {}
        for recipe_id in recipe_ids:
            global_counts = self._global_counts[recipe_id]
            score = self._score_counter(global_counts, weight=0.4)
            if user_id:
                score += self._score_counter(self._user_counts[user_id][recipe_id], weight=0.6)
            boosts[recipe_id] = max(-1.0, min(1.0, score))
        return boosts

    async def store_feedback(self, user_id: str, feedback: FeedbackRequest) -> bool:
        self._global_counts[feedback.recipe_id][feedback.action] += 1
        self._user_counts[user_id][feedback.recipe_id][feedback.action] += 1
        return True

    async def get_profile(self, user_id: str) -> UserProfile:
        return self._profiles.get(user_id, UserProfile(user_id=user_id))

    async def upsert_profile(self, profile: UserProfile) -> UserProfile:
        self._profiles[profile.user_id] = profile
        return profile

    @staticmethod
    def _score_counter(counter: Counter[str], weight: float) -> float:
        positive = sum(counter[action] for action in POSITIVE_ACTIONS)
        negative = sum(counter[action] for action in NEGATIVE_ACTIONS)
        total = positive + negative
        if total == 0:
            return 0.0
        return weight * ((positive - negative) / total)


class SupabaseFeedbackStore(FeedbackStore):
    provider = "supabase"

    def __init__(self, settings: Settings) -> None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise ValueError("Supabase settings are required")
        self.base_url = settings.supabase_url.rstrip("/")
        self.headers = {
            "apikey": settings.supabase_service_role_key,
            "Authorization": f"Bearer {settings.supabase_service_role_key}",
            "Content-Type": "application/json",
        }
        self._profile_fallback: dict[str, UserProfile] = {}

    async def get_boosts(self, user_id: str | None, recipe_ids: list[int]) -> dict[int, float]:
        if not recipe_ids:
            return {}

        recipe_filter = ",".join(str(recipe_id) for recipe_id in recipe_ids)
        params = {
            "select": "recipe_id,user_id,action,created_at",
            "recipe_id": f"in.({recipe_filter})",
            "limit": "1000",
        }

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{self.base_url}/rest/v1/recipe_feedback", headers=self.headers, params=params
            )
            if response.status_code >= 400:
                return {recipe_id: 0.0 for recipe_id in recipe_ids}
            rows = response.json()

        global_counts: dict[int, Counter[str]] = defaultdict(Counter)
        user_counts: dict[int, Counter[str]] = defaultdict(Counter)
        for row in rows:
            recipe_id = int(row["recipe_id"])
            action = row["action"]
            global_counts[recipe_id][action] += 1
            if user_id and row.get("user_id") == user_id:
                user_counts[recipe_id][action] += 1

        boosts: dict[int, float] = {}
        for recipe_id in recipe_ids:
            score = MemoryFeedbackStore._score_counter(global_counts[recipe_id], 0.4)
            score += MemoryFeedbackStore._score_counter(user_counts[recipe_id], 0.6)
            boosts[recipe_id] = max(-1.0, min(1.0, score))
        return boosts

    async def store_feedback(self, user_id: str, feedback: FeedbackRequest) -> bool:
        payload = {
            "user_id": user_id,
            "recipe_id": feedback.recipe_id,
            "action": feedback.action,
            "reason": feedback.reason,
            "comment": feedback.comment,
            "context": feedback.context,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.base_url}/rest/v1/recipe_feedback",
                headers={**self.headers, "Prefer": "return=minimal"},
                json=payload,
            )
            response.raise_for_status()
            if feedback.action == "save":
                saved_payload = {
                    "user_id": user_id,
                    "recipe_id": feedback.recipe_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                saved_response = await client.post(
                    f"{self.base_url}/rest/v1/saved_recipes",
                    headers={**self.headers, "Prefer": "resolution=merge-duplicates,return=minimal"},
                    params={"on_conflict": "user_id,recipe_id"},
                    json=saved_payload,
                )
                if saved_response.status_code != 404:
                    saved_response.raise_for_status()
        return True

    async def store_recommendation_event(
        self, user_id: str | None, request_payload: dict[str, Any], recipe_ids: list[int]
    ) -> None:
        payload = {
            "user_id": user_id,
            "request_payload": request_payload,
            "recipe_ids": recipe_ids,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.base_url}/rest/v1/recommendation_events",
                headers={**self.headers, "Prefer": "return=minimal"},
                json=payload,
            )
            response.raise_for_status()

    async def get_profile(self, user_id: str) -> UserProfile:
        params = {
            "select": "user_id,dietary_preference,pantry_ingredients,labels",
            "user_id": f"eq.{user_id}",
            "limit": "1",
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{self.base_url}/rest/v1/user_preferences", headers=self.headers, params=params
            )
            if response.status_code == 404:
                return self._profile_fallback.get(user_id, UserProfile(user_id=user_id))
            response.raise_for_status()
            rows = response.json()
        if not rows:
            return UserProfile(user_id=user_id)
        row = rows[0]
        return UserProfile(
            user_id=user_id,
            dietary_preference=row.get("dietary_preference"),
            pantry_ingredients=row.get("pantry_ingredients") or [],
            labels=row.get("labels") or {},
        )

    async def upsert_profile(self, profile: UserProfile) -> UserProfile:
        payload = {
            "user_id": profile.user_id,
            "dietary_preference": profile.dietary_preference,
            "pantry_ingredients": profile.pantry_ingredients,
            "labels": profile.labels,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.base_url}/rest/v1/user_preferences",
                headers={**self.headers, "Prefer": "resolution=merge-duplicates,return=representation"},
                params={"on_conflict": "user_id"},
                json=payload,
            )
            if response.status_code == 404:
                self._profile_fallback[profile.user_id] = profile
                return profile
            response.raise_for_status()
            rows = response.json()
        if rows:
            row = rows[0]
            return UserProfile(
                user_id=user_id_from_row(row, profile.user_id),
                dietary_preference=row.get("dietary_preference"),
                pantry_ingredients=row.get("pantry_ingredients") or [],
                labels=row.get("labels") or {},
            )
        return profile


def user_id_from_row(row: dict[str, Any], fallback: str) -> str:
    value = row.get("user_id")
    return str(value) if value else fallback


def build_feedback_store(settings: Settings) -> FeedbackStore:
    if settings.supabase_url and settings.supabase_service_role_key:
        return SupabaseFeedbackStore(settings)
    return MemoryFeedbackStore()
