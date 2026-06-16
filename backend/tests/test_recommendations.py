import os

os.environ["GARDEN_SUPABASE_URL"] = ""
os.environ["GARDEN_SUPABASE_SERVICE_ROLE_KEY"] = ""

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.recommender import collapse_compound_ingredients, estimate_hidden_prep_minutes, infer_ingredients_from_text


client = TestClient(app)


def test_metadata_is_non_empty():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["cuisines"]
    assert data["diets"]
    assert data["courses"]
    assert data["ingredients"]


def test_vegan_request_only_returns_vegan_recipes():
    response = client.post(
        "/recommendations",
        json={"cuisine": "Indian", "diet": "Vegan", "course": "Dinner", "ingredients": ["tofu", "spinach"]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert {recipe["diet"] for recipe in results} == {"Vegan"}


def test_gluten_free_request_only_returns_gluten_free_recipes():
    response = client.post(
        "/recommendations",
        json={"cuisine": "Mexican", "diet": "Gluten Free", "course": "Lunch", "ingredients": ["rice", "beans"]},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert {recipe["diet"] for recipe in results} == {"Gluten Free"}


def test_ingredient_heavy_query_ranks_ingredient_matches():
    response = client.post(
        "/recommendations",
        json={
            "cuisine": "Mexican",
            "diet": "Vegetarian",
            "course": "Lunch",
            "ingredients": ["rice", "beans", "spinach", "tomato"],
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["score"]["ingredient_coverage"] > 0


def test_feedback_requires_authenticated_user():
    response = client.post("/feedback", json={"recipe_id": 1, "action": "like"})
    assert response.status_code == 401


def test_feedback_accepts_user_id():
    response = client.post("/feedback", json={"recipe_id": 1, "action": "like", "user_id": "test-user"})
    assert response.status_code == 200
    assert response.json()["stored"] is True


def test_profile_can_store_pantry_and_diet():
    payload = {
        "user_id": "profile-test",
        "dietary_preference": "Vegan",
        "pantry_ingredients": ["rice", "lentils"],
        "labels": {"rice": "always in kitchen"},
    }
    response = client.put("/profile/profile-test", json=payload)
    assert response.status_code == 200
    saved = response.json()["profile"]
    assert saved["dietary_preference"] == "Vegan"
    assert saved["pantry_ingredients"] == ["rice", "lentils"]

    response = client.get("/profile/profile-test")
    assert response.status_code == 200
    assert response.json()["profile"]["labels"]["rice"] == "always in kitchen"


def test_vegetarian_recipe_with_actual_egg_is_repaired_to_eggetarian():
    response = client.get("/recipes/11")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Veg Chili Cheese Burgers Recipe"
    assert data["diet"] == "Eggetarian"
    assert "egg" in data["ingredients"]
    assert "sunfloweroil" in data["ingredients"]
    assert "sunflower" not in data["ingredients"]

    response = client.post(
        "/recommendations",
        json={"diet": "Vegetarian", "fridge_ingredients": ["egg"], "limit": 50},
    )
    assert response.status_code == 200
    assert all(recipe["recipe_id"] != 11 for recipe in response.json()["results"])


def test_serving_suggestion_chicken_does_not_change_vegetarian_diet():
    response = client.get("/recipes/44")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Herbal Brown Rice Recipe"
    assert data["diet"] == "Vegetarian"
    assert data["hidden_prep_time_mins"] == 0
    assert "chicken" not in data["ingredients"]


def test_recommendations_do_not_repeat_duplicate_recipe_names():
    response = client.post("/recommendations", json={"limit": 50})
    assert response.status_code == 200
    names = [recipe["name"].lower() for recipe in response.json()["results"]]
    assert len(names) == len(set(names))


def test_recommendation_uses_pantry_and_fridge_buckets():
    response = client.post(
        "/recommendations",
        json={
            "cuisine": "Indian",
            "diet": "Vegetarian",
            "course": "Dinner",
            "pantry_ingredients": ["rice", "lentils"],
            "fridge_ingredients": ["spinach", "tomato"],
            "unavailable_ingredients": ["tomato"],
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert "tomato" in results[0]["score"]["missing_ingredients"] or "tomato" not in results[0]["score"]["matched_ingredients"]


def test_order_intent_returns_missing_items_without_real_provider_credentials():
    response = client.post(
        "/order-intent",
        json={"user_id": "test-user", "recipe_id": 1, "provider": "blinkit", "missing_ingredients": ["milk", "paneer"]},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "integration_required"
    assert data["items"] == ["milk", "paneer"]


def test_compound_ingredients_prefer_specific_token():
    assert collapse_compound_ingredients(["coconut", "milk", "coconutmilk"]) == ["coconutmilk"]
    assert collapse_compound_ingredients(["cheese", "cheddarcheese", "rice"]) == ["cheddarcheese", "rice"]


def test_surprise_request_does_not_need_cuisine_or_course():
    response = client.post(
        "/recommendations",
        json={
            "diet": "Vegetarian",
            "fridge_ingredients": ["spinach", "tomato"],
            "liked_recipe_ids": [1, 2],
            "surprise": True,
        },
    )
    assert response.status_code == 200
    assert response.json()["results"]


def test_selected_cuisine_is_ranked_before_backfill_when_available():
    response = client.post(
        "/recommendations",
        json={
            "cuisine": "Mexican",
            "diet": "Vegetarian",
            "course": "Lunch",
            "pantry_ingredients": ["rice", "lentils", "wheat", "flour", "salt", "turmeric"],
            "fridge_ingredients": ["tomato", "spinach"],
            "limit": 5,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert results[0]["cuisine"] == "Mexican"
    first_alternate = next((index for index, recipe in enumerate(results) if recipe["score"]["alternate_cuisine"]), None)
    if first_alternate is not None:
        assert all(recipe["cuisine"] == "Mexican" for recipe in results[:first_alternate])


def test_broad_breakfast_matches_dataset_breakfast_courses():
    response = client.post(
        "/recommendations",
        json={
            "cuisine": "Italian Recipes",
            "diet": "Vegetarian",
            "course": "Breakfast",
            "pantry_ingredients": ["flour", "salt", "spinach", "wheat"],
            "limit": 3,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert results[0]["course"] in {"Breakfast", "Indian Breakfast", "North Indian Breakfast", "South Indian Breakfast", "World Breakfast"}
    assert results[0]["score"]["course_match"] == 1.0


def test_long_soaking_time_is_added_to_planning_time():
    assert estimate_hidden_prep_minutes("Soak rajma in water for 8 to 10 hours.") == 600
    assert estimate_hidden_prep_minutes("Marinate the paneer overnight before cooking.") == 480
    assert estimate_hidden_prep_minutes("Use sprouted rajma and brown chickpeas for the filling.") == 480
    assert estimate_hidden_prep_minutes("Soy chunks soaked in hot water for 15 minutes, green beans chopped.") == 0
    assert estimate_hidden_prep_minutes("Soak the brown rice in 2-1 / 2 cups of water for 1/2 hour.") == 0


def test_time_filter_uses_effective_total_time():
    response = client.post(
        "/recommendations",
        json={
            "diet": "High Protein Vegetarian",
            "course": "Lunch",
            "pantry_ingredients": ["kidneybeans", "brownchickpea", "lentils"],
            "fridge_ingredients": ["tomato", "spinach"],
            "max_total_time_mins": 60,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert all(recipe["effective_total_time_mins"] <= 60 for recipe in results)
    assert all(recipe["score"]["match_label"] for recipe in results)


def test_soya_chunks_are_recovered_from_source_ingredient_text():
    assert infer_ingredients_from_text("1 cup Soy Chunks (Nuggets) - cooked in water") == ["soyachunks"]
    response = client.get("/recipes/5682")
    assert response.status_code == 200
    assert "soyachunks" in response.json()["ingredients"]


def test_unsupported_artifact_ingredients_are_removed_from_recipe_detail():
    response = client.get("/recipes/4041")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Baked Stuffed Baati With Churma And Panchratna Dal Recipe"
    assert "emu" not in data["ingredients"]
    assert "agar" not in data["ingredients"]
    assert "wholewheatflour" in data["ingredients"]
    assert "semolina" in data["ingredients"]


def test_eggplant_does_not_create_egg_ingredient_or_eggetarian_diet():
    response = client.get("/recipes/104")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Bengaluru Style Brinjal Gravy Recipe"
    assert "eggplant" in data["ingredients"]
    assert "egg" not in data["ingredients"]
    assert data["diet"] == "Vegetarian"


def test_vegan_recipe_with_paneer_is_repaired_to_vegetarian():
    response = client.get("/recipes/3531")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Paneer Carrot Stuffed Paratha Recipe"
    assert "paneer" in data["ingredients"]
    assert data["diet"] == "Vegetarian"


def test_short_soy_biryani_soak_does_not_become_overnight_prep():
    response = client.get("/recipes/142")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Hyderabad Soy Biryani Recipe With Vegetables & Palak"
    assert data["hidden_prep_time_mins"] == 0
    assert data["effective_total_time_mins"] == 90


def test_fresh_ingredients_are_ranked_before_pantry_only_matches():
    response = client.post(
        "/recommendations",
        json={
            "cuisine": "Indian",
            "diet": "Vegetarian",
            "course": "Lunch",
            "pantry_ingredients": ["rice", "lentils", "wheat", "flour", "salt", "turmeric"],
            "fridge_ingredients": ["tomato", "spinach", "paneer"],
            "limit": 12,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    first_pantry_only = next(
        (index for index, recipe in enumerate(results) if "paneer" not in recipe["score"]["matched_ingredients"]),
        None,
    )
    if first_pantry_only is not None:
        assert all("paneer" in recipe["score"]["matched_ingredients"] for recipe in results[:first_pantry_only])


def test_single_fresh_paneer_request_returns_paneer_recipes_first():
    response = client.post(
        "/recommendations",
        json={
            "cuisine": "Indian",
            "diet": "Vegetarian",
            "course": "Lunch",
            "pantry_ingredients": ["rice", "lentils", "wheat", "flour", "salt", "turmeric"],
            "fridge_ingredients": ["paneer"],
            "limit": 4,
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results
    assert all("paneer" in recipe["score"]["matched_ingredients"] for recipe in results[:4])
