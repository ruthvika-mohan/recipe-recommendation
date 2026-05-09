import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { createClient } from "@supabase/supabase-js";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;
const supabase = SUPABASE_URL && SUPABASE_ANON_KEY ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY) : null;

const DEFAULT_FORM = {
  cuisine: "",
  course: "",
  maxTimeMins: "",
  fridgeIngredients: [],
};

const DEFAULT_PROFILE = {
  dietary_preference: "Vegetarian",
  pantry_ingredients: ["rice", "lentils", "wheat", "flour", "salt", "turmeric"],
  labels: {},
};

function icon(name, extra = "") {
  return <span className={`material-symbols-outlined ${extra}`}>{name}</span>;
}

function demoUserId() {
  const key = "garden-guide-demo-user";
  const existing = localStorage.getItem(key);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(key, id);
  return id;
}

function localProfileKey(userId) {
  return `garden-guide-profile-${userId}`;
}

function readLocalProfile(userId) {
  const local = localStorage.getItem(localProfileKey(userId));
  if (!local) return null;
  try {
    return normalizeProfile(JSON.parse(local), DEFAULT_PROFILE);
  } catch {
    return null;
  }
}

function readFallbackLocalProfile(userId) {
  const direct = readLocalProfile(userId);
  if (direct) return direct;
  const demoId = localStorage.getItem("garden-guide-demo-user");
  if (demoId && demoId !== userId) return readLocalProfile(demoId);
  return null;
}

function hasUsefulProfile(profile) {
  return Boolean(
    profile?.dietary_preference ||
      profile?.pantry_ingredients?.length ||
      Object.keys(profile?.labels || {}).length,
  );
}

function normalizeProfile(profile, fallback = DEFAULT_PROFILE) {
  return {
    user_id: profile?.user_id || fallback?.user_id,
    dietary_preference: profile?.dietary_preference || fallback?.dietary_preference || null,
    pantry_ingredients: profile?.pantry_ingredients?.length
      ? profile.pantry_ingredients
      : fallback?.pantry_ingredients || [],
    labels: { ...(fallback?.labels || {}), ...(profile?.labels || {}) },
  };
}

function App() {
  const [metadata, setMetadata] = useState({ cuisines: [], diets: [], courses: [], ingredients: [] });
  const [form, setForm] = useState(DEFAULT_FORM);
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const [pantryInput, setPantryInput] = useState("");
  const [fridgeInput, setFridgeInput] = useState("");
  const [unavailableIngredients, setUnavailableIngredients] = useState([]);
  const [orderIntent, setOrderIntent] = useState(null);
  const [results, setResults] = useState([]);
  const [likedRecipes, setLikedRecipes] = useState([]);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [profileOpen, setProfileOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [session, setSession] = useState(null);

  const userId = session?.user?.id || demoUserId();
  const pantrySuggestions = useIngredientSuggestions(pantryInput, metadata.ingredients);
  const fridgeSuggestions = useIngredientSuggestions(fridgeInput, metadata.ingredients);
  const availableIngredients = useMemo(
    () =>
      [...profile.pantry_ingredients, ...form.fridgeIngredients].filter(
        (item) => !unavailableIngredients.includes(item),
      ),
    [profile.pantry_ingredients, form.fridgeIngredients, unavailableIngredients],
  );
  const selectedRecipeMissing = useMemo(() => {
    if (!selectedRecipe) return [];
    const available = new Set(availableIngredients);
    return selectedRecipe.ingredients.filter((ingredient) => !available.has(ingredient));
  }, [availableIngredients, selectedRecipe]);

  useEffect(() => {
    loadMetadata();
    loadProfile(userId);
    loadLikedRecipes(userId);
    if (!supabase) return;
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session);
      if (data.session?.user?.id) loadProfile(data.session.user.id);
      if (data.session?.user?.id) loadLikedRecipes(data.session.user.id);
    });
    const { data: listener } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      setSession(nextSession);
      if (nextSession?.user?.id) loadProfile(nextSession.user.id);
      if (nextSession?.user?.id) loadLikedRecipes(nextSession.user.id);
    });
    return () => listener.subscription.unsubscribe();
  }, [userId]);

  function useIngredientSuggestions(input, ingredients) {
    const query = input.trim().toLowerCase();
    if (!query) return [];
    return ingredients.filter((item) => item.toLowerCase().includes(query)).slice(0, 8);
  }

  async function loadMetadata() {
    try {
      const response = await fetch(`${API_BASE_URL}/metadata`);
      if (!response.ok) {
        throw new Error(`Metadata failed to load from ${API_BASE_URL}/metadata (${response.status})`);
      }
      const data = await response.json();
      setMetadata(data);
    } catch (nextError) {
      setError(`Load failed: ${nextError.message}`);
    }
  }

  async function signIn() {
    if (!supabase) {
      setError("Add VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable real sign in. Demo identity is active.");
      return;
    }
    const email = window.prompt("Email for magic link sign in");
    if (!email) return;
    const { error: signInError } = await supabase.auth.signInWithOtp({ email });
    if (signInError) setError(signInError.message);
    else setError("Check your email for the magic link.");
  }

  async function signOut() {
    if (supabase) await supabase.auth.signOut();
  }

  function updateForm(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function updateProfile(key, value) {
    setProfile((current) => ({ ...current, [key]: value }));
  }

  async function loadProfile(nextUserId) {
    const localProfile = readFallbackLocalProfile(nextUserId);
    if (localProfile) setProfile(localProfile);
    try {
      const response = await fetch(`${API_BASE_URL}/profile/${nextUserId}`);
      if (!response.ok) return;
      const data = await response.json();
      const remoteProfile = normalizeProfile(data.profile, null);
      const nextProfile = hasUsefulProfile(remoteProfile)
        ? normalizeProfile(remoteProfile, localProfile || DEFAULT_PROFILE)
        : localProfile || DEFAULT_PROFILE;
      setProfile(nextProfile);
      setProfileOpen(!nextProfile.pantry_ingredients.length);
      localStorage.setItem(localProfileKey(nextUserId), JSON.stringify({ user_id: nextUserId, ...nextProfile }));
    } catch {
      // Local profile fallback is enough for development.
    }
  }

  async function saveProfile() {
    const nextProfile = { user_id: userId, ...profile };
    localStorage.setItem(localProfileKey(userId), JSON.stringify(nextProfile));
    try {
      const response = await fetch(`${API_BASE_URL}/profile/${userId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-User-Id": userId },
        body: JSON.stringify(nextProfile),
      });
      if (!response.ok) throw new Error(`Profile save failed (${response.status})`);
      setError("");
      setNotice("Kitchen profile saved.");
      setProfileOpen(false);
    } catch (nextError) {
      setError(`Saved locally. Backend profile save failed: ${nextError.message}`);
    }
  }

  function addIngredient(value, bucket) {
    const normalized = value.trim().toLowerCase();
    if (!normalized) return;
    if (bucket === "pantry") {
      if (!profile.pantry_ingredients.includes(normalized)) {
        updateProfile("pantry_ingredients", [...profile.pantry_ingredients, normalized]);
      }
      setPantryInput("");
    } else {
      if (!form.fridgeIngredients.includes(normalized)) {
        updateForm("fridgeIngredients", [...form.fridgeIngredients, normalized]);
      }
      setFridgeInput("");
    }
  }

  function removeIngredient(value, bucket) {
    if (bucket === "pantry") {
      updateProfile(
        "pantry_ingredients",
        profile.pantry_ingredients.filter((item) => item !== value),
      );
      return;
    }
    updateForm(
      "fridgeIngredients",
      form.fridgeIngredients.filter((item) => item !== value),
    );
  }

  function toggleUnavailable(value) {
    setUnavailableIngredients((current) =>
      current.includes(value) ? current.filter((item) => item !== value) : [...current, value],
    );
  }

  async function markRecipeIngredient(value) {
    const normalized = value.trim().toLowerCase();
    if (!normalized) return;
    const alreadyAvailable = availableIngredients.includes(normalized);
    if (alreadyAvailable) {
      setUnavailableIngredients((current) => (current.includes(normalized) ? current : [...current, normalized]));
      setNotice(`${normalized} marked unavailable for this recipe.`);
      return;
    }

    const nextProfile = {
      ...profile,
      pantry_ingredients: profile.pantry_ingredients.includes(normalized)
        ? profile.pantry_ingredients
        : [...profile.pantry_ingredients, normalized],
      labels: { ...profile.labels, [normalized]: "always in kitchen" },
    };
    setProfile(nextProfile);
    setUnavailableIngredients((current) => current.filter((item) => item !== normalized));
    localStorage.setItem(localProfileKey(userId), JSON.stringify({ user_id: userId, ...nextProfile }));
    setNotice(`${normalized} saved to your monthly kitchen setup.`);
    try {
      await fetch(`${API_BASE_URL}/profile/${userId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json", "X-User-Id": userId },
        body: JSON.stringify({ user_id: userId, ...nextProfile }),
      });
    } catch {
      // Local pantry save still works; the next setup save can retry backend persistence.
    }
  }

  async function recommend() {
    return requestRecommendations({
      cuisine: form.cuisine || null,
      diet: profile.dietary_preference,
      course: form.course || null,
      max_total_time_mins: form.maxTimeMins ? Number(form.maxTimeMins) : null,
    });
  }

  async function surpriseMe() {
    return requestRecommendations({
      cuisine: null,
      diet: profile.dietary_preference,
      course: null,
      max_total_time_mins: form.maxTimeMins ? Number(form.maxTimeMins) : null,
      liked_recipe_ids: likedRecipes.map((recipe) => recipe.recipe_id),
      surprise: true,
    });
  }

  async function requestRecommendations(extraPayload) {
    setLoading(true);
    setError("");
    setSelectedRecipe(null);
    setOrderIntent(null);
    try {
      const response = await fetch(`${API_BASE_URL}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...extraPayload,
          pantry_ingredients: profile.pantry_ingredients,
          fridge_ingredients: form.fridgeIngredients,
          unavailable_ingredients: unavailableIngredients,
          user_id: userId,
          limit: 12,
        }),
      });
      if (!response.ok) throw new Error(`Recommendation request failed (${response.status})`);
      const data = await response.json();
      setResults(data.results);
    } catch (nextError) {
      setError(nextError.message);
    } finally {
      setLoading(false);
    }
  }

  async function loadRecipe(recipeId) {
    setDetailLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/recipes/${recipeId}`);
      if (!response.ok) throw new Error(`Recipe details failed to load (${response.status})`);
      setSelectedRecipe(await response.json());
    } catch (nextError) {
      setError(nextError.message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function sendFeedback(recipe, action) {
    try {
      const token = session?.access_token;
      const headers = { "Content-Type": "application/json", "X-User-Id": userId };
      if (token) headers.Authorization = `Bearer ${token}`;
      const response = await fetch(`${API_BASE_URL}/feedback`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          recipe_id: recipe.recipe_id,
          action,
          user_id: userId,
          context: {
            cuisine: form.cuisine,
            diet: profile.dietary_preference,
            course: form.course,
            pantry_ingredients: profile.pantry_ingredients,
            fridge_ingredients: form.fridgeIngredients,
          },
        }),
      });
      if (!response.ok) throw new Error(`Feedback could not be saved (${response.status})`);
      setResults((current) =>
        current.map((item) => (item.recipe_id === recipe.recipe_id ? { ...item, lastFeedback: action } : item)),
      );
      if (["like", "save", "cooked"].includes(action)) {
        setLikedRecipes((current) => {
          const next = [recipe, ...current.filter((item) => item.recipe_id !== recipe.recipe_id)].slice(0, 12);
          localStorage.setItem(`garden-guide-liked-${userId}`, JSON.stringify(next));
          return next;
        });
      }
    } catch (nextError) {
      setError(nextError.message);
    }
  }

  function loadLikedRecipes(nextUserId) {
    const local = localStorage.getItem(`garden-guide-liked-${nextUserId}`);
    if (local) setLikedRecipes(JSON.parse(local));
  }

  async function createOrderIntent(provider = "any") {
    if (!selectedRecipe) return;
    try {
      const response = await fetch(`${API_BASE_URL}/order-intent`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-User-Id": userId },
        body: JSON.stringify({
          user_id: userId,
          recipe_id: selectedRecipe.recipe_id,
          provider,
          missing_ingredients: selectedRecipeMissing,
        }),
      });
      if (!response.ok) throw new Error(`Order intent failed (${response.status})`);
      setOrderIntent(await response.json());
    } catch (nextError) {
      setError(nextError.message);
    }
  }

  return (
    <div className="min-h-screen bg-background text-on-surface pb-24">
      <TopBar session={session} onSignIn={signIn} onSignOut={signOut} />
      <main className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-4 py-8">
        <section className="grid items-center gap-8 lg:grid-cols-[1fr_430px]">
          <div>
            <div className="mb-4 inline-flex rounded-full bg-secondary-container px-4 py-1 text-xs font-semibold uppercase tracking-wide text-on-secondary-container">
              Waste less, cook better
            </div>
            <h1 className="font-display text-4xl font-bold leading-tight text-primary md:text-6xl">
              Find recipes that actually match your kitchen.
            </h1>
            <p className="mt-4 max-w-2xl text-lg leading-8 text-on-surface-variant">
              Diet is enforced first, ingredients drive the score, and cuisine becomes a preference instead of a
              shortcut.
            </p>
          </div>
          <SearchPanel
            form={form}
            profile={profile}
            metadata={metadata}
            pantryInput={pantryInput}
            fridgeInput={fridgeInput}
            pantrySuggestions={pantrySuggestions}
            fridgeSuggestions={fridgeSuggestions}
            unavailableIngredients={unavailableIngredients}
            profileOpen={profileOpen}
            loading={loading}
            onPantryInput={setPantryInput}
            onFridgeInput={setFridgeInput}
            onAddIngredient={addIngredient}
            onRemoveIngredient={removeIngredient}
            onToggleUnavailable={toggleUnavailable}
            onUpdate={updateForm}
            onProfileUpdate={updateProfile}
            onSaveProfile={saveProfile}
            onEditProfile={() => setProfileOpen(true)}
            onSubmit={recommend}
          />
        </section>

        {error && <div className="rounded-lg border border-error/20 bg-red-50 px-4 py-3 text-sm text-error">{error}</div>}
        {notice && !error && (
          <div className="rounded-lg border border-secondary-container bg-secondary-container/50 px-4 py-3 text-sm font-semibold text-primary">
            {notice}
          </div>
        )}

        <section className="grid gap-6 lg:grid-cols-[1fr_420px]">
          <ResultsGrid results={results} loading={loading} onOpen={loadRecipe} onFeedback={sendFeedback} />
          <div className="space-y-6">
            <LikedRecipes recipes={likedRecipes} onOpen={loadRecipe} />
            <RecipeDetailPanel
              recipe={selectedRecipe}
              loading={detailLoading}
              availableIngredients={availableIngredients}
              missingIngredients={selectedRecipeMissing}
              orderIntent={orderIntent}
              onMarkIngredient={markRecipeIngredient}
              onOrder={createOrderIntent}
            />
          </div>
        </section>
      </main>
      <BottomNav />
    </div>
  );
}

function TopBar({ session, onSignIn, onSignOut }) {
  return (
    <header className="sticky top-0 z-50 border-b border-outline-variant/40 bg-surface/85 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <span className="rounded-full bg-secondary-container p-2 text-primary">{icon("restaurant")}</span>
          <span className="font-display text-2xl font-semibold text-primary">Garden Guide</span>
        </div>
        <div className="flex items-center gap-2">
          <button className="icon-button" aria-label="Search">
            {icon("search")}
          </button>
          <button onClick={session ? onSignOut : onSignIn} className="icon-button" aria-label="Account">
            {icon(session ? "logout" : "account_circle")}
          </button>
        </div>
      </div>
    </header>
  );
}

function SearchPanel({
  form,
  profile,
  metadata,
  pantryInput,
  fridgeInput,
  pantrySuggestions,
  fridgeSuggestions,
  unavailableIngredients,
  profileOpen,
  loading,
  onPantryInput,
  onFridgeInput,
  onAddIngredient,
  onRemoveIngredient,
  onToggleUnavailable,
  onUpdate,
  onProfileUpdate,
  onSaveProfile,
  onEditProfile,
  onSubmit,
}) {
  return (
    <section className="rounded-xl border border-outline-variant bg-surface-container-lowest p-5 shadow-sm">
      <div className="mb-5 rounded-xl bg-secondary-container/70 p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="font-display text-lg font-semibold text-primary">Monthly kitchen setup</h2>
            <p className="mt-1 text-sm text-on-surface-variant">
              Save staples and diet once. These are treated as always in your kitchen.
            </p>
          </div>
          {profileOpen ? (
            <button onClick={onSaveProfile} className="rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-on-primary">
              Save
            </button>
          ) : (
            <button onClick={onEditProfile} className="rounded-lg bg-white px-3 py-2 text-sm font-semibold text-primary">
              Edit setup
            </button>
          )}
        </div>

        {profileOpen ? (
          <>
            <div className="mt-4">
              <label className="label">Dietary preference</label>
              <div className="mt-2 flex flex-wrap gap-2">
                {metadata.diets.map((diet) => (
                  <button
                    key={diet}
                    onClick={() => onProfileUpdate("dietary_preference", diet)}
                    className={`chip ${profile.dietary_preference === diet ? "chip-selected" : ""}`}
                  >
                    {diet}
                  </button>
                ))}
              </div>
            </div>

            <IngredientBucket
              label="Always in your kitchen"
              description="Monthly staples like rice, lentils, wheat, flour, salt, spices."
              bucket="pantry"
              items={profile.pantry_ingredients}
              input={pantryInput}
              suggestions={pantrySuggestions}
              unavailableIngredients={unavailableIngredients}
              onInput={onPantryInput}
              onAddIngredient={onAddIngredient}
              onRemoveIngredient={onRemoveIngredient}
              onToggleUnavailable={onToggleUnavailable}
            />
          </>
        ) : (
          <div className="mt-4">
            <div className="text-sm font-semibold text-primary">{profile.dietary_preference || "No diet saved"}</div>
            <div className="mt-2 flex flex-wrap gap-2">
              {profile.pantry_ingredients.map((item) => (
                <span key={item} className="rounded-lg bg-primary px-3 py-1 text-sm text-on-primary">
                  {item} <span className="ml-1 rounded bg-white/20 px-1 text-[10px] uppercase">always</span>
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Select
          label="Cuisine"
          value={form.cuisine}
          options={[{ label: "Any cuisine", value: "" }, ...metadata.cuisines]}
          onChange={(value) => onUpdate("cuisine", value)}
        />
        <Select
          label="Meal"
          value={form.course}
          options={[{ label: "Any meal", value: "" }, ...metadata.courses]}
          onChange={(value) => onUpdate("course", value)}
        />
        <Select
          label="Time available"
          value={form.maxTimeMins}
          options={[
            { label: "Any time", value: "" },
            { label: "30 minutes", value: "30" },
            { label: "45 minutes", value: "45" },
            { label: "1 hour", value: "60" },
            { label: "2 hours", value: "120" },
            { label: "Same day", value: "360" },
            { label: "Overnight okay", value: "900" },
          ]}
          onChange={(value) => onUpdate("maxTimeMins", value)}
        />
      </div>

      <IngredientBucket
        label="What's fresh today?"
        description="Perishable fridge items like milk, paneer, curd, cheese, vegetables, herbs."
        bucket="fridge"
        items={form.fridgeIngredients}
        input={fridgeInput}
        suggestions={fridgeSuggestions}
        unavailableIngredients={unavailableIngredients}
        onInput={onFridgeInput}
        onAddIngredient={onAddIngredient}
        onRemoveIngredient={onRemoveIngredient}
        onToggleUnavailable={onToggleUnavailable}
      />

      <div className="mt-6">
        <button onClick={onSubmit} disabled={loading} className="flex w-full items-center justify-center gap-2 rounded-full bg-primary px-6 py-4 font-display text-lg font-semibold text-on-primary shadow-lg transition hover:bg-tertiary disabled:opacity-60">
          {loading ? "Finding..." : "Generate Recipes"}
          {icon("auto_awesome")}
        </button>
      </div>
    </section>
  );
}

function IngredientBucket({
  label,
  description,
  bucket,
  items,
  input,
  suggestions,
  unavailableIngredients,
  onInput,
  onAddIngredient,
  onRemoveIngredient,
  onToggleUnavailable,
}) {
  return (
    <div className="mt-5">
      <label className="label">{label}</label>
      <p className="mt-1 px-1 text-xs text-on-surface-variant">{description}</p>
      <div className="mt-2 rounded-xl bg-surface-container-low p-3 focus-within:ring-2 focus-within:ring-secondary">
        <div className="mb-2 flex flex-wrap gap-2">
          {items.map((ingredient) => {
            const unavailable = unavailableIngredients.includes(ingredient);
            return (
              <span
                key={ingredient}
                className={`inline-flex items-center gap-1 rounded-lg px-3 py-1 text-sm ${
                  unavailable ? "bg-surface-container-high text-on-surface-variant line-through" : "bg-primary text-on-primary"
                }`}
              >
                {ingredient}
                {bucket === "pantry" && <span className="rounded bg-white/20 px-1 text-[10px] uppercase">always</span>}
                <button onClick={() => onToggleUnavailable(ingredient)} aria-label={`Toggle availability for ${ingredient}`}>
                  {icon(unavailable ? "add_circle" : "remove_circle", "text-base")}
                </button>
                <button onClick={() => onRemoveIngredient(ingredient, bucket)} aria-label={`Remove ${ingredient}`}>
                  {icon("close", "text-base")}
                </button>
              </span>
            );
          })}
        </div>
        <input
          value={input}
          onChange={(event) => onInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              onAddIngredient(input, bucket);
            }
          }}
          className="w-full border-0 bg-transparent p-0 text-on-surface placeholder:text-outline focus:ring-0"
          placeholder={bucket === "pantry" ? "Add monthly staple..." : "Add fresh ingredient..."}
        />
      </div>
      {suggestions.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {suggestions.map((suggestion) => (
            <button key={suggestion} onClick={() => onAddIngredient(suggestion, bucket)} className="suggestion-chip">
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Select({ label, value, options, onChange }) {
  return (
    <label>
      <span className="label">{label}</span>
      <span className="relative mt-2 block">
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className="w-full appearance-none rounded-xl border-0 bg-surface-container-low px-4 py-3 text-on-surface focus:ring-2 focus:ring-secondary"
        >
          {options.map((option) => {
            const nextValue = typeof option === "string" ? option : option.value;
            const nextLabel = typeof option === "string" ? option : option.label;
            return (
            <option key={nextValue} value={nextValue}>
              {nextLabel}
            </option>
            );
          })}
        </select>
        {icon("expand_more", "pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-outline")}
      </span>
    </label>
  );
}

function ResultsGrid({ results, loading, onOpen, onFeedback }) {
  if (loading) {
    return <section className="rounded-xl bg-surface-container-low p-8 text-on-surface-variant">Checking what recipe ingredients you already have...</section>;
  }
  if (!results.length) {
    return (
      <section className="rounded-xl border border-dashed border-outline-variant bg-surface-container-low p-8 text-on-surface-variant">
        Select your preferences and generate recipes to see ranked matches.
      </section>
    );
  }
  return (
    <section>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h2 className="font-display text-2xl font-semibold text-primary">Top Recommended Recipes</h2>
          <p className="text-sm text-on-surface-variant">Ranked by how much of each recipe you can already make, then cuisine and learning signals.</p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {results.map((recipe) => (
          <RecipeCard key={recipe.recipe_id} recipe={recipe} onOpen={onOpen} onFeedback={onFeedback} />
        ))}
      </div>
    </section>
  );
}

function LikedRecipes({ recipes, onOpen }) {
  return (
    <aside className="rounded-xl border border-outline-variant bg-surface-container-lowest p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="font-display text-xl font-semibold text-primary">Liked recipes</h2>
          <p className="mt-1 text-sm text-on-surface-variant">Quickly revisit recipes you liked or saved.</p>
        </div>
      </div>
      <div className="mt-4 space-y-2">
        {recipes.length ? (
          recipes.slice(0, 5).map((recipe) => (
            <button
              key={recipe.recipe_id}
              onClick={() => onOpen(recipe.recipe_id)}
              className="w-full rounded-lg bg-surface-container-low px-3 py-2 text-left text-sm text-on-surface-variant hover:bg-secondary-container"
            >
              <span className="block font-semibold text-primary">{recipe.name}</span>
              <span>{recipe.cuisine} · {recipe.diet}</span>
            </button>
          ))
        ) : (
          <p className="text-sm text-on-surface-variant">Like recipes to save them here.</p>
        )}
      </div>
    </aside>
  );
}

function RecipeCard({ recipe, onOpen, onFeedback }) {
  return (
    <article className="overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest shadow-sm transition hover:shadow-md">
      <div className="flex h-32 items-end bg-[linear-gradient(135deg,#d8e6d9,#cde5ff)] p-4">
        <div className="rounded-full bg-white/90 px-3 py-1 text-sm font-semibold text-primary">
          {recipe.score.match_label}
        </div>
      </div>
      <div className="p-4">
        <div className="mb-3 flex flex-wrap gap-2">
          <span className="meta-pill">{recipe.cuisine}</span>
          <span className="meta-pill">{recipe.course}</span>
          <span className="meta-pill">{recipe.diet}</span>
          {recipe.score.alternate_cuisine && <span className="meta-pill-alt">alternate cuisine</span>}
          {recipe.score.alternate_course && <span className="meta-pill-alt">alternate meal</span>}
        </div>
        <h3 className="min-h-16 font-display text-lg font-semibold text-primary">{recipe.name}</h3>
        <p className="mt-2 text-sm text-on-surface-variant">{recipe.score.match_reason}</p>
        <FitLine label="Already have" value={recipe.score.recipe_coverage} />
        <div className="mt-3 text-xs text-on-surface-variant">
          Have: {recipe.score.matched_ingredients.length ? recipe.score.matched_ingredients.join(", ") : "No recipe ingredients yet"}
        </div>
        <div className="mt-1 text-xs text-on-surface-variant">
          Need: {recipe.score.missing_ingredients.length ? recipe.score.missing_ingredients.slice(0, 8).join(", ") : "Nothing else"}
          {recipe.score.missing_ingredients.length > 8 ? ` +${recipe.score.missing_ingredients.length - 8} more` : ""}
        </div>
        <div className="mt-2 text-xs font-semibold text-primary">{formatRecipeTime(recipe)}</div>
        <div className="mt-4 grid grid-cols-[1fr_auto_auto_auto] gap-2">
          <button onClick={() => onOpen(recipe.recipe_id)} className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-on-primary">
            Learn More
          </button>
          <button onClick={() => onFeedback(recipe, "like")} className="small-button" aria-label="Like">
            {icon("thumb_up")}
          </button>
          <button onClick={() => onFeedback(recipe, "save")} className="small-button" aria-label="Save">
            {icon("bookmark")}
          </button>
          <button onClick={() => onFeedback(recipe, "not_relevant")} className="small-button" aria-label="Not relevant">
            {icon("thumb_down")}
          </button>
        </div>
        {recipe.lastFeedback && <p className="mt-2 text-xs font-semibold text-primary">Feedback saved: {recipe.lastFeedback}</p>}
      </div>
    </article>
  );
}

function fitWord(value) {
  if (value >= 0.7) return "strong";
  if (value >= 0.35) return "some";
  return "low";
}

function FitLine({ label, value }) {
  return (
    <div className="mt-3">
      <div className="mb-1 flex justify-between text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
        <span>{label}</span>
        <span>{fitWord(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-surface-container-high">
        <div className="h-full rounded-full bg-primary" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

function formatRecipeTime(recipe) {
  const total = recipe.effective_total_time_mins ?? recipe.total_time_mins;
  if (!total) return "Time not listed";
  const hidden = recipe.hidden_prep_time_mins || 0;
  const base = total >= 60 ? `${Math.floor(total / 60)}h ${total % 60 ? `${total % 60}m` : ""}`.trim() : `${total}m`;
  return hidden ? `Plan for ${base} including soaking/resting` : `Ready in ${base}`;
}

function RecipeDetailPanel({
  recipe,
  loading,
  availableIngredients,
  missingIngredients,
  orderIntent,
  onMarkIngredient,
  onOrder,
}) {
  if (loading) return <aside className="rounded-xl bg-surface-container-low p-6 text-on-surface-variant">Loading details...</aside>;
  if (!recipe) {
    return (
      <aside className="rounded-xl border border-outline-variant bg-surface-container-lowest p-6">
        <div className="mb-4 rounded-xl bg-secondary-container p-4 text-primary">{icon("eco", "text-4xl")}</div>
        <h2 className="font-display text-2xl font-semibold text-primary">Recipe details</h2>
        <p className="mt-2 text-on-surface-variant">Open a result to see ingredients, cooking steps, timing, and source link.</p>
      </aside>
    );
  }
  return (
    <aside className="sticky top-24 max-h-[calc(100vh-8rem)] overflow-auto rounded-xl border border-outline-variant bg-surface-container-lowest p-6 shadow-sm">
      <div className="mb-3 flex flex-wrap gap-2">
        <span className="meta-pill">{recipe.cuisine}</span>
        <span className="meta-pill">{recipe.course}</span>
        <span className="meta-pill">{recipe.diet}</span>
      </div>
      <h2 className="font-display text-2xl font-semibold text-primary">{recipe.name}</h2>
      <div className="mt-4 grid grid-cols-2 gap-2 text-center text-sm sm:grid-cols-4">
        <Info label="Prep" value={`${recipe.prep_time_mins ?? 0}m`} />
        <Info label="Cook" value={`${recipe.cook_time_mins ?? 0}m`} />
        <Info label="Serves" value={recipe.servings ?? "-"} />
        <Info label="Plan for" value={formatRecipeTime(recipe).replace("Plan for ", "").replace("Ready in ", "")} />
      </div>
      {recipe.time_note && (
        <p className="mt-3 rounded-lg bg-secondary-container/60 px-3 py-2 text-sm text-primary">{recipe.time_note}</p>
      )}
      <h3 className="mt-6 font-display text-lg font-semibold text-primary">Ingredients</h3>
      <div className="mt-3 flex flex-wrap gap-2">
        {recipe.ingredients.map((ingredient) => {
          const available = availableIngredients.includes(ingredient);
          return (
            <button
              key={ingredient}
              onClick={() => onMarkIngredient(ingredient)}
              className={`rounded-lg px-3 py-1 text-sm ${
                available ? "bg-secondary-container text-primary" : "bg-surface-container text-on-surface-variant"
              }`}
              title={available ? "Click if you do not have this for this recipe" : "Click to save this to your pantry"}
            >
              {available ? "I have this" : "I need this"} · {ingredient}
            </button>
          );
        })}
      </div>
      <div className="mt-5 rounded-xl border border-outline-variant bg-surface-container-low p-4">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="font-display text-lg font-semibold text-primary">Missing items</h3>
            <p className="mt-1 text-sm text-on-surface-variant">
              Mark ingredients you do not have. Use this list for a future Blinkit/Swiggy Instamart cart.
            </p>
          </div>
          <button
            onClick={() => onOrder("any")}
            disabled={!missingIngredients.length}
            className="rounded-lg bg-primary px-3 py-2 text-sm font-semibold text-on-primary disabled:opacity-50"
          >
            Order list
          </button>
        </div>
        <div className="mt-3 flex flex-wrap gap-2">
          {missingIngredients.length ? (
            missingIngredients.map((ingredient) => (
              <span key={ingredient} className="rounded-lg bg-white px-3 py-1 text-sm text-on-surface-variant">
                {ingredient}
              </span>
            ))
          ) : (
            <span className="text-sm text-on-surface-variant">You seem to have everything needed.</span>
          )}
        </div>
        {orderIntent && (
          <div className="mt-3 rounded-lg bg-white p-3 text-sm text-on-surface-variant">
            <p className="font-semibold text-primary">{orderIntent.provider} order intent</p>
            <p>{orderIntent.message}</p>
            {orderIntent.search_url && (
              <a href={orderIntent.search_url} target="_blank" rel="noreferrer" className="mt-2 inline-flex font-semibold text-primary">
                Search missing items
              </a>
            )}
          </div>
        )}
      </div>
      <h3 className="mt-6 font-display text-lg font-semibold text-primary">How to prepare</h3>
      <ol className="mt-3 space-y-4">
        {recipe.instructions.map((step, index) => (
          <li key={`${index}-${step.slice(0, 12)}`} className="flex gap-3">
            <span className="flex h-8 w-8 flex-none items-center justify-center rounded-full bg-primary text-sm font-bold text-on-primary">
              {index + 1}
            </span>
            <p className="text-sm leading-6 text-on-surface-variant">{step}</p>
          </li>
        ))}
      </ol>
      {recipe.url && (
        <a href={recipe.url} target="_blank" rel="noreferrer" className="mt-6 inline-flex items-center gap-2 text-sm font-semibold text-primary">
          Open original recipe {icon("open_in_new", "text-base")}
        </a>
      )}
    </aside>
  );
}

function Info({ label, value }) {
  return (
    <div className="rounded-lg bg-surface-container-low p-3">
      <div className="text-xs uppercase tracking-wide text-on-surface-variant">{label}</div>
      <div className="font-display text-lg font-semibold text-primary">{value}</div>
    </div>
  );
}

function BottomNav() {
  return (
    <nav className="fixed bottom-0 left-0 z-50 flex w-full justify-around rounded-t-xl bg-surface px-4 pb-4 pt-2 shadow-[0_-4px_20px_rgba(22,52,34,0.08)] md:hidden">
      {["home", "kitchen", "restaurant_menu", "bookmark"].map((name) => (
        <button key={name} className="flex flex-col items-center rounded-full px-4 py-1 text-on-surface-variant">
          {icon(name)}
        </button>
      ))}
    </nav>
  );
}

createRoot(document.getElementById("root")).render(<App />);
