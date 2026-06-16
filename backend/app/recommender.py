from __future__ import annotations

import ast
import math
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .feedback import FeedbackStore
from .schemas import (
    MetadataResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecipeDetail,
    RecipeSummary,
    ScoreBreakdown,
)


TOKEN_RE = re.compile(r"[^a-z0-9]+")
LONG_PREP_RE = re.compile(
    r"\b(?:soak|soaked|rest|rested|ferment|fermented|marinate|marinated|sprout|sprouted)\b[^.]{0,90}?"
    r"(?:(?P<overnight>overnight)|(?P<first>\d+(?:\.\d+)?)\s*(?:-|to)?\s*(?P<second>\d+(?:\.\d+)?)?\s*(?P<unit>hours?|hrs?|minutes?|mins?))",
    re.IGNORECASE,
)
SPROUTED_LEGUME_RE = re.compile(
    r"\b(?:sprouted|sprouts?)\b[^.]{0,80}\b(?:rajma|kidney\s*beans?|chickpeas?|chana|moong|lentils?|dal|beans?)\b"
    r"|\b(?:rajma|kidney\s*beans?|chickpeas?|chana|moong|lentils?|dal|beans?)\b[^.]{0,80}\b(?:sprouted|sprouts?)\b",
    re.IGNORECASE,
)
INGREDIENT_TEXT_ALIASES = [
    (re.compile(r"\bsoy(?:a)?\s+(?:chunks?|nuggets?|balls?)\b", re.IGNORECASE), "soyachunks"),
    (re.compile(r"\bcoconut\s+milk\b", re.IGNORECASE), "coconutmilk"),
    (re.compile(r"\bcoconut\s+oil\b", re.IGNORECASE), "coconutoil"),
    (re.compile(r"\bpeanut\s+butter\b", re.IGNORECASE), "peanutbutter"),
    (re.compile(r"\bginger\s+garlic\s+paste\b", re.IGNORECASE), "gingergarlicpaste"),
    (re.compile(r"\bwhole\s+wheat\s+flou?r\b", re.IGNORECASE), "wholewheatflour"),
    (re.compile(r"\bgram\s+flou?r\b", re.IGNORECASE), "gramflour"),
    (re.compile(r"\brice\s+flou?r\b", re.IGNORECASE), "riceflour"),
    (re.compile(r"\bgaram\s+masala\b", re.IGNORECASE), "garammasala"),
    (re.compile(r"\bred\s+chilli?\s+powder\b", re.IGNORECASE), "redchillipowder"),
    (re.compile(r"\bblack\s+pepper\s+powder\b", re.IGNORECASE), "blackpepperpowder"),
    (re.compile(r"\btomato\s+puree\b", re.IGNORECASE), "tomatopuree"),
    (re.compile(r"\bparmesan\s+cheese\b", re.IGNORECASE), "parmesancheese"),
    (re.compile(r"\bmozzarella\s+cheese\b", re.IGNORECASE), "mozzarellacheese"),
    (re.compile(r"\bcheddar\s+cheese\b", re.IGNORECASE), "cheddarcheese"),
    (re.compile(r"\bbasmati\s+rice\b", re.IGNORECASE), "basmatirice"),
    (re.compile(r"\b(?:amchur|dry\s+mango\s+powder)\b", re.IGNORECASE), "drymangopowder"),
    (re.compile(r"\bsunflower\s+oil\b", re.IGNORECASE), "sunfloweroil"),
]
SERVING_CUE_RE = re.compile(
    r"\bserve\b[^.]{0,160}?\b(?:with|along with|as|after|hot along with)\b",
    re.IGNORECASE,
)
EGG_RE = re.compile(r"\b(?:whole\s+)?eggs?\b", re.IGNORECASE)
DAIRY_OR_HONEY_INGREDIENTS = {
    "butter",
    "buttermilk",
    "cheddarcheese",
    "cheese",
    "condensedmilk",
    "cottagecheese",
    "cream",
    "creamcheese",
    "curd",
    "ghee",
    "greekyogurt",
    "honey",
    "milk",
    "mozzarellacheese",
    "paneer",
    "parmesancheese",
    "ricottacheese",
    "yogurt",
}
SOURCE_ARTIFACT_PATTERNS = {
    "agar": re.compile(r"(?<![a-z])agar(?:\s+agar)?s?(?![a-z])", re.IGNORECASE),
    "corn": re.compile(r"(?<![a-z])corns?(?![a-z])", re.IGNORECASE),
    "egg": re.compile(r"(?<![a-z])eggs?(?![a-z])", re.IGNORECASE),
    "emu": re.compile(r"(?<![a-z])emus?(?![a-z])", re.IGNORECASE),
    "pear": re.compile(r"(?<![a-z])pears?(?![a-z])", re.IGNORECASE),
    "stew": re.compile(r"(?<![a-z])stews?(?![a-z])", re.IGNORECASE),
    "tea": re.compile(r"(?<![a-z])teas?(?![a-z])", re.IGNORECASE),
}
ARTIFACT_PRONE_INGREDIENTS = set(SOURCE_ARTIFACT_PATTERNS)

DIET_ALIASES = {
    "Vegetarian": {"Vegetarian", "High Protein Vegetarian", "Vegan"},
    "High Protein Vegetarian": {"High Protein Vegetarian", "Vegetarian"},
    "Non Vegeterian": {"Non Vegeterian", "High Protein Non Vegetarian"},
    "High Protein Non Vegetarian": {"High Protein Non Vegetarian", "Non Vegeterian"},
}

COURSE_ALIASES = {
    "Breakfast": {"Breakfast", "Indian Breakfast", "North Indian Breakfast", "South Indian Breakfast", "World Breakfast"},
    "Lunch": {"Lunch", "Main Course", "One Pot Dish"},
    "Dinner": {"Dinner", "Main Course", "One Pot Dish"},
    "Main Course": {"Main Course", "Lunch", "Dinner", "One Pot Dish"},
    "Snack": {"Snack", "Appetizer"},
    "Appetizer": {"Appetizer", "Snack"},
}


def normalize_text(value: object) -> str:
    return TOKEN_RE.sub("", str(value).lower())


def normalize_query(value: object) -> str:
    return " ".join(str(value).lower().split())


def parse_ingredients(value: object) -> list[str]:
    if isinstance(value, list):
        raw_items = value
    else:
        try:
            raw_items = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            raw_items = str(value).split(",")
    return collapse_compound_ingredients([normalize_text(item) for item in raw_items if normalize_text(item)])


def source_ingredient_text(row: pd.Series) -> str:
    return " ".join(
        str(row.get(column, ""))
        for column in ("TranslatedRecipeName", "TranslatedIngredients")
        if pd.notna(row.get(column, ""))
    )


def is_supported_artifact_ingredient(ingredient: str, row: pd.Series) -> bool:
    pattern = SOURCE_ARTIFACT_PATTERNS.get(ingredient)
    if pattern is None:
        return True
    return bool(pattern.search(source_ingredient_text(row)))


def remove_unsupported_artifacts(items: list[str], row: pd.Series) -> list[str]:
    return [
        item
        for item in items
        if item not in ARTIFACT_PRONE_INGREDIENTS or is_supported_artifact_ingredient(item, row)
    ]


def infer_ingredients_from_text(*values: object) -> list[str]:
    text = " ".join(str(value) for value in values if pd.notna(value))
    inferred = [ingredient for pattern, ingredient in INGREDIENT_TEXT_ALIASES if pattern.search(text)]
    return sorted(set(inferred))


def parse_recipe_ingredients(row: pd.Series) -> list[str]:
    parsed = remove_unsupported_artifacts(parse_ingredients(row.get("FinalIngredientList")), row)
    inferred = infer_ingredients_from_text(
        row.get("TranslatedRecipeName"),
        row.get("TranslatedIngredients"),
        row.get("TranslatedInstructions"),
    )
    return collapse_compound_ingredients([*parsed, *inferred])


def core_recipe_text(value: object) -> str:
    text = str(value)
    match = SERVING_CUE_RE.search(text)
    return text[: match.start()] if match else text


def has_actual_egg(row: pd.Series) -> bool:
    if "egg" in set(row.get("_parsed_ingredients", [])):
        return True
    source_text = " ".join(
        str(row.get(column, ""))
        for column in ("TranslatedRecipeName", "TranslatedIngredients")
    )
    source_text = f"{source_text} {core_recipe_text(row.get('TranslatedInstructions', ''))}"
    return bool(EGG_RE.search(source_text))


def repaired_diet(row: pd.Series) -> str:
    diet = str(row.get("Diet", "")).replace("\ufeff", "").strip()
    if diet in {"Vegetarian", "High Protein Vegetarian"} and has_actual_egg(row):
        return "Eggetarian"
    if diet == "Vegan" and set(row.get("_parsed_ingredients", [])) & DAIRY_OR_HONEY_INGREDIENTS:
        return "Vegetarian"
    return diet


def collapse_compound_ingredients(items: list[str]) -> list[str]:
    unique_items = sorted(set(items), key=lambda item: (-len(item), item))
    kept: list[str] = []
    for item in unique_items:
        if any(item != other and item in other for other in kept):
            continue
        kept.append(item)
    return sorted(kept)


def split_instructions(value: object) -> list[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)(\s+|\s*$)", str(value))
    return [sentence.strip() for sentence in sentences if sentence and sentence.strip()]


def estimate_hidden_prep_minutes(*values: object) -> int:
    text = " ".join(str(value) for value in values if pd.notna(value)).lower()
    estimates: list[int] = []
    for match in LONG_PREP_RE.finditer(text):
        matched_text = match.group(0)
        if "/" in matched_text or " cup" in matched_text:
            continue
        if match.group("overnight"):
            estimates.append(8 * 60)
            continue
        amount = float(match.group("second") or match.group("first") or 0)
        unit = match.group("unit") or ""
        minutes = int(amount * 60) if unit.startswith(("hour", "hr")) else int(amount)
        if minutes >= 60:
            estimates.append(minutes)
    if not estimates and SPROUTED_LEGUME_RE.search(text):
        estimates.append(8 * 60)
    return max(estimates, default=0)


def build_time_note(hidden_minutes: int) -> str | None:
    if not hidden_minutes:
        return None
    hours = hidden_minutes // 60
    minutes = hidden_minutes % 60
    if hours and minutes:
        estimate = f"{hours}h {minutes}m"
    elif hours:
        estimate = f"{hours}h"
    else:
        estimate = f"{minutes}m"
    return f"Includes about {estimate} of soaking, resting, fermenting, or marinating time found in the recipe steps."


class RecipeRecommender:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        cleaned_catalog = project_dir / "data" / "final_ingredient_list_cleaned.csv"
        source_catalog = cleaned_catalog if cleaned_catalog.exists() else project_dir / "data" / "final_ingredient_list_created.csv"
        self.recipes = pd.read_csv(source_catalog).reset_index(drop=True)
        self.ingredients = pd.read_csv(project_dir / "list_ingredients_for_app.csv")["0"].dropna().unique().tolist()

        with open(project_dir / "concatenated_features.pkl", "rb") as file:
            self.features = pickle.load(file)
        with open(project_dir / "vectorizer.pkl", "rb") as file:
            self.vectorizer = pickle.load(file)

        self.recipes["recipe_id"] = self.recipes.index.astype(int)
        self.recipes["_recipe_display_name"] = self.recipes.apply(self._recipe_name, axis=1)
        self.recipes["_normalized_cuisine"] = self.recipes["Cuisine"].map(normalize_text)
        self.recipes["_normalized_course"] = self.recipes["Course"].map(normalize_text)
        self.recipes["_parsed_ingredients"] = self.recipes.apply(parse_recipe_ingredients, axis=1)
        self.recipes["_ingredient_set"] = self.recipes["_parsed_ingredients"].map(set)
        ingredient_counts: dict[str, int] = {}
        for ingredient_set in self.recipes["_ingredient_set"]:
            for ingredient in ingredient_set:
                ingredient_counts[ingredient] = ingredient_counts.get(ingredient, 0) + 1
        self._ingredient_signal_weights = {
            ingredient: 1 / math.sqrt(count) for ingredient, count in ingredient_counts.items() if count
        }
        self.recipes["_effective_diet"] = self.recipes.apply(repaired_diet, axis=1)
        self.recipes["_normalized_diet"] = self.recipes["_effective_diet"].map(normalize_text)
        if "IsDuplicateRecipeName" in self.recipes.columns:
            self.recipes["_is_primary_duplicate"] = ~self.recipes["IsDuplicateRecipeName"].fillna(False).astype(bool)
        else:
            self.recipes["_is_primary_duplicate"] = ~self.recipes["_recipe_display_name"].str.lower().duplicated(keep="first")
        self.recipes["_hidden_prep_time_mins"] = self.recipes.apply(
            lambda row: estimate_hidden_prep_minutes(
                row.get("TranslatedRecipeName"),
                row.get("TranslatedIngredients"),
                row.get("TranslatedInstructions"),
            ),
            axis=1,
        )
        numeric_total = pd.to_numeric(self.recipes.get("TotalTimeInMins"), errors="coerce").fillna(0).astype(int)
        self.recipes["_effective_total_time_mins"] = numeric_total + self.recipes["_hidden_prep_time_mins"]
        self.recipes["_time_note"] = self.recipes["_hidden_prep_time_mins"].map(build_time_note)

    def metadata(self) -> MetadataResponse:
        return MetadataResponse(
            cuisines=self._unique_values("Cuisine"),
            diets=self._unique_values("_effective_diet"),
            courses=self._unique_values("Course"),
            ingredients=sorted({str(item) for item in self.ingredients}),
        )

    async def recommend(
        self, request: RecommendationRequest, feedback_store: FeedbackStore
    ) -> RecommendationResponse:
        candidates = self._recommendable_candidates(self._strict_diet_candidates(request.diet))
        strict_diet_applied = bool(request.diet)

        if candidates.empty:
            return RecommendationResponse(results=[], total_candidates=0, strict_diet_applied=strict_diet_applied)

        primary_candidates = self._primary_course_candidates(candidates, request.course)
        alternate_candidates = self._alternate_course_candidates(candidates, request.course, primary_candidates.index)
        ranked_primary = await self._rank_bucketed_by_cuisine(
            primary_candidates, request, feedback_store, alternate_course=False
        )

        ranked = ranked_primary
        if len(ranked_primary) < min(6, request.limit) and not alternate_candidates.empty:
            ranked_alternate = await self._rank_bucketed_by_cuisine(
                alternate_candidates, request, feedback_store, alternate_course=True
            )
            ranked = [*ranked_primary, *ranked_alternate]

        ranked = ranked[: request.limit]
        await feedback_store.store_recommendation_event(
            request.user_id,
            request.model_dump(),
            [recipe.recipe_id for recipe in ranked],
        )
        return RecommendationResponse(
            results=ranked,
            total_candidates=int(len(candidates)),
            strict_diet_applied=strict_diet_applied,
        )

    def recipe_detail(self, recipe_id: int) -> RecipeDetail:
        if recipe_id < 0 or recipe_id >= len(self.recipes):
            raise KeyError(recipe_id)
        row = self.recipes.iloc[recipe_id]
        return RecipeDetail(
            recipe_id=recipe_id,
            name=self._recipe_name(row),
            cuisine=str(row["Cuisine"]),
            course=str(row["Course"]),
            diet=str(row["_effective_diet"]),
            prep_time_mins=self._optional_int(row.get("PrepTimeInMins")),
            cook_time_mins=self._optional_int(row.get("CookTimeInMins")),
            total_time_mins=self._optional_int(row.get("TotalTimeInMins")),
            hidden_prep_time_mins=self._optional_int(row.get("_hidden_prep_time_mins")),
            effective_total_time_mins=self._optional_int(row.get("_effective_total_time_mins")),
            time_note=str(row["_time_note"]) if row.get("_time_note") else None,
            servings=self._optional_int(row.get("Servings")),
            ingredients=list(row["_parsed_ingredients"]),
            instructions=split_instructions(row["TranslatedInstructions"]),
            url=str(row["URL"]) if pd.notna(row.get("URL")) else None,
        )

    async def _rank_candidates(
        self,
        candidates: pd.DataFrame,
        request: RecommendationRequest,
        feedback_store: FeedbackStore,
        alternate_course: bool,
        alternate_cuisine: bool,
    ) -> list[RecipeSummary]:
        if candidates.empty:
            return []

        candidates = self._time_candidates(candidates, request.max_total_time_mins)
        if candidates.empty:
            return []

        fresh_ingredients = self._fresh_ingredients(request)
        if fresh_ingredients:
            fresh_match_counts = candidates["_ingredient_set"].map(lambda recipe_items: len(fresh_ingredients & recipe_items))
            fresh_candidates = candidates[fresh_match_counts > 0]
            pantry_only_candidates = candidates[fresh_match_counts == 0]
            if not fresh_candidates.empty:
                ranked_fresh = await self._rank_candidates_unbucketed(
                    fresh_candidates,
                    request,
                    feedback_store,
                    alternate_course,
                    alternate_cuisine,
                )
                if len(ranked_fresh) >= min(6, request.limit) or pantry_only_candidates.empty:
                    return ranked_fresh
                ranked_pantry = await self._rank_candidates_unbucketed(
                    pantry_only_candidates,
                    request,
                    feedback_store,
                    alternate_course,
                    alternate_cuisine,
                )
                return [*ranked_fresh, *ranked_pantry]

        return await self._rank_candidates_unbucketed(
            candidates,
            request,
            feedback_store,
            alternate_course,
            alternate_cuisine,
        )

    async def _rank_candidates_unbucketed(
        self,
        candidates: pd.DataFrame,
        request: RecommendationRequest,
        feedback_store: FeedbackStore,
        alternate_course: bool,
        alternate_cuisine: bool,
    ) -> list[RecipeSummary]:
        if candidates.empty:
            return []

        query = self._query_text(request)
        user_vector = self.vectorizer.transform([query])
        candidates = self._candidate_window(candidates, request)
        candidate_indices = candidates.index.to_list()
        similarities = cosine_similarity(user_vector, self.features[candidate_indices])[0]
        recipe_ids = candidates["recipe_id"].astype(int).to_list()
        feedback_boosts = await feedback_store.get_boosts(request.user_id, recipe_ids)
        liked_recipes = set(request.liked_recipe_ids)

        selected_ingredients = self._available_ingredients(request)
        requested_cuisine = normalize_text(request.cuisine) if request.cuisine else ""
        requested_course = normalize_text(request.course) if request.course else ""

        results: list[RecipeSummary] = []
        for similarity, (_, row) in zip(similarities, candidates.iterrows()):
            recipe_id = int(row["recipe_id"])
            recipe_ingredients = set(row["_ingredient_set"])
            matched_set = selected_ingredients & recipe_ingredients
            matched = sorted(matched_set)
            ingredient_coverage = len(matched_set) / len(selected_ingredients) if selected_ingredients else 0.0
            recipe_coverage = len(matched_set) / len(recipe_ingredients) if recipe_ingredients else 0.0
            cuisine_match = 1.0 if requested_cuisine and row["_normalized_cuisine"] == requested_cuisine else 0.0
            course_match = 1.0 if requested_course and self._course_matches(row, request.course) else 0.0
            fresh_signal = self._fresh_signal_score(request, recipe_ingredients)
            feedback_boost = feedback_boosts.get(recipe_id, 0.0)
            if request.surprise and liked_recipes:
                liked_rows = self.recipes[self.recipes["recipe_id"].isin(liked_recipes)]
                cuisine_overlap = row["_normalized_cuisine"] in set(liked_rows["_normalized_cuisine"])
                diet_overlap = row["_normalized_diet"] in set(liked_rows["_normalized_diet"])
                feedback_boost += 0.3 if cuisine_overlap else 0.0
                feedback_boost += 0.2 if diet_overlap else 0.0
                if recipe_id in liked_recipes:
                    feedback_boost -= 0.8
            final_score = (
                0.25 * recipe_coverage
                + 0.45 * fresh_signal
                + 0.15 * float(similarity)
                + 0.15 * cuisine_match
                + 0.10 * course_match
                + 0.05 * feedback_boost
            )
            match_label, match_reason = self._match_label(
                final_score=final_score,
                recipe_coverage=recipe_coverage,
                matched_count=len(matched_set),
                total_recipe_ingredients=len(recipe_ingredients),
                alternate_course=alternate_course,
            )
            missing_recipe_ingredients = sorted(recipe_ingredients - selected_ingredients)

            results.append(
                RecipeSummary(
                    recipe_id=recipe_id,
                    name=self._recipe_name(row),
                    cuisine=str(row["Cuisine"]),
                    course=str(row["Course"]),
                    diet=str(row["_effective_diet"]),
                    prep_time_mins=self._optional_int(row.get("PrepTimeInMins")),
                    cook_time_mins=self._optional_int(row.get("CookTimeInMins")),
                    total_time_mins=self._optional_int(row.get("TotalTimeInMins")),
                    hidden_prep_time_mins=self._optional_int(row.get("_hidden_prep_time_mins")),
                    effective_total_time_mins=self._optional_int(row.get("_effective_total_time_mins")),
                    time_note=str(row["_time_note"]) if row.get("_time_note") else None,
                    servings=self._optional_int(row.get("Servings")),
                    ingredients=list(row["_parsed_ingredients"]),
                    url=str(row["URL"]) if pd.notna(row.get("URL")) else None,
                    score=ScoreBreakdown(
                        final_score=round(final_score, 4),
                        match_label=match_label,
                        match_reason=match_reason,
                        ingredient_coverage=round(ingredient_coverage, 4),
                        recipe_coverage=round(recipe_coverage, 4),
                        tfidf_similarity=round(float(similarity), 4),
                        cuisine_match=cuisine_match,
                        course_match=course_match,
                        feedback_boost=round(feedback_boost, 4),
                        matched_ingredients=matched,
                        missing_ingredients=missing_recipe_ingredients,
                        alternate_course=alternate_course,
                        alternate_cuisine=alternate_cuisine,
                    ),
                )
            )
        return sorted(results, key=lambda recipe: recipe.score.final_score, reverse=True)

    async def _rank_bucketed_by_cuisine(
        self,
        candidates: pd.DataFrame,
        request: RecommendationRequest,
        feedback_store: FeedbackStore,
        alternate_course: bool,
    ) -> list[RecipeSummary]:
        if candidates.empty:
            return []
        if not request.cuisine:
            return await self._rank_candidates(candidates, request, feedback_store, alternate_course, False)

        normalized_cuisine = normalize_text(request.cuisine)
        cuisine_candidates = candidates[candidates["_normalized_cuisine"] == normalized_cuisine]
        other_candidates = candidates[candidates["_normalized_cuisine"] != normalized_cuisine]
        ranked_cuisine = await self._rank_candidates(
            cuisine_candidates, request, feedback_store, alternate_course, False
        )

        if len(ranked_cuisine) >= min(6, request.limit) or other_candidates.empty:
            return ranked_cuisine

        ranked_other = await self._rank_candidates(other_candidates, request, feedback_store, alternate_course, True)
        return [*ranked_cuisine, *ranked_other]

    def _strict_diet_candidates(self, diet: str | None) -> pd.DataFrame:
        if not diet:
            return self.recipes
        allowed = DIET_ALIASES.get(diet, {diet})
        normalized_allowed = {normalize_text(item) for item in allowed}
        return self.recipes[self.recipes["_normalized_diet"].isin(normalized_allowed)]

    @staticmethod
    def _recommendable_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
        return candidates[candidates["_is_primary_duplicate"]]

    @staticmethod
    def _time_candidates(candidates: pd.DataFrame, max_total_time_mins: int | None) -> pd.DataFrame:
        if not max_total_time_mins:
            return candidates
        return candidates[candidates["_effective_total_time_mins"] <= max_total_time_mins]

    def _primary_course_candidates(self, candidates: pd.DataFrame, course: str | None) -> pd.DataFrame:
        if not course:
            return candidates
        allowed = self._course_match_set(course)
        primary = candidates[candidates["_normalized_course"].isin(allowed)]
        return primary if not primary.empty else candidates.iloc[0:0]

    def _alternate_course_candidates(
        self, candidates: pd.DataFrame, course: str | None, primary_indices: pd.Index
    ) -> pd.DataFrame:
        if not course:
            return candidates.iloc[0:0]
        allowed = self._course_match_set(course)
        alternate = candidates[
            (~candidates["_normalized_course"].isin(allowed)) & (~candidates.index.isin(primary_indices))
        ]
        if alternate.empty:
            alternate = candidates[~candidates.index.isin(primary_indices)]
        return alternate

    @staticmethod
    def _course_match_set(course: str | None) -> set[str]:
        if not course:
            return set()
        return {normalize_text(course), *(normalize_text(item) for item in COURSE_ALIASES.get(course, set()))}

    def _course_matches(self, row: pd.Series, course: str | None) -> bool:
        return bool(course and row["_normalized_course"] in self._course_match_set(course))

    def _fresh_signal_score(self, request: RecommendationRequest, recipe_ingredients: set[str]) -> float:
        fresh_ingredients = self._fresh_ingredients(request)
        if not fresh_ingredients:
            return 0.0
        total_weight = sum(self._ingredient_signal_weights.get(item, 1.0) for item in fresh_ingredients)
        if not total_weight:
            return 0.0
        matched_weight = sum(
            self._ingredient_signal_weights.get(item, 1.0) for item in fresh_ingredients & recipe_ingredients
        )
        return matched_weight / total_weight

    def _query_text(self, request: RecommendationRequest) -> str:
        parts = [request.cuisine, request.diet, request.course, " ".join(sorted(self._available_ingredients(request)))]
        return normalize_query(" ".join(part for part in parts if part))

    def _candidate_window(self, candidates: pd.DataFrame, request: RecommendationRequest) -> pd.DataFrame:
        selected = self._available_ingredients(request)
        if not selected:
            return candidates.head(300)
        coverage = candidates["_ingredient_set"].map(lambda recipe_items: len(selected & recipe_items))
        return candidates.assign(_coverage_count=coverage).sort_values("_coverage_count", ascending=False).head(300)

    @staticmethod
    def _match_label(
        final_score: float,
        recipe_coverage: float,
        matched_count: int,
        total_recipe_ingredients: int,
        alternate_course: bool,
    ) -> tuple[str, str]:
        if final_score >= 0.55 or recipe_coverage >= 0.7:
            label = "Highly recommended"
        elif final_score >= 0.32 or recipe_coverage >= 0.35:
            label = "Good match"
        else:
            label = "This could work"

        if total_recipe_ingredients:
            missing_count = max(total_recipe_ingredients - matched_count, 0)
            reason = f"You already have {matched_count} of {total_recipe_ingredients} recipe ingredients"
            if missing_count:
                reason = f"{reason}; {missing_count} to buy or skip"
        elif alternate_course:
            reason = "Diet fits, but this is an alternate meal type"
        else:
            reason = "Diet fits, with a weaker pantry match"
        return label, reason

    @staticmethod
    def _available_ingredients(request: RecommendationRequest) -> set[str]:
        combined = [*request.ingredients, *request.pantry_ingredients, *request.fridge_ingredients]
        unavailable = {normalize_text(item) for item in request.unavailable_ingredients if normalize_text(item)}
        return set(collapse_compound_ingredients([normalize_text(item) for item in combined if normalize_text(item)])) - unavailable

    @staticmethod
    def _fresh_ingredients(request: RecommendationRequest) -> set[str]:
        unavailable = {normalize_text(item) for item in request.unavailable_ingredients if normalize_text(item)}
        return set(
            collapse_compound_ingredients(
                [normalize_text(item) for item in request.fridge_ingredients if normalize_text(item)]
            )
        ) - unavailable

    def _unique_values(self, column: str) -> list[str]:
        values = {str(value).replace("\ufeff", "").strip() for value in self.recipes[column].dropna().unique()}
        return sorted(value for value in values if value)

    @staticmethod
    def _recipe_name(row: pd.Series) -> str:
        return str(row["TranslatedRecipeName"]).split(" - ")[0].strip()

    @staticmethod
    def _optional_int(value: object) -> int | None:
        if pd.isna(value):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
