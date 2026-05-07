import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { createClient } from "@supabase/supabase-js";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_ANON_KEY = import.meta.env.VITE_SUPABASE_ANON_KEY;
const supabase = SUPABASE_URL && SUPABASE_ANON_KEY ? createClient(SUPABASE_URL, SUPABASE_ANON_KEY) : null;

const DEFAULT_FORM = {
  cuisine: "Indian",
  diet: "Vegetarian",
  course: "Dinner",
  ingredients: ["rice", "tomato", "spinach"],
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

function App() {
  const [metadata, setMetadata] = useState({ cuisines: [], diets: [], courses: [], ingredients: [] });
  const [form, setForm] = useState(DEFAULT_FORM);
  const [ingredientInput, setIngredientInput] = useState("");
  const [results, setResults] = useState([]);
  const [selectedRecipe, setSelectedRecipe] = useState(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [session, setSession] = useState(null);

  const userId = session?.user?.id || demoUserId();
  const suggestions = useMemo(() => {
    const query = ingredientInput.trim().toLowerCase();
    if (!query) return [];
    return metadata.ingredients.filter((item) => item.toLowerCase().includes(query)).slice(0, 8);
  }, [ingredientInput, metadata.ingredients]);

  useEffect(() => {
    loadMetadata();
    if (!supabase) return;
    supabase.auth.getSession().then(({ data }) => setSession(data.session));
    const { data: listener } = supabase.auth.onAuthStateChange((_event, nextSession) => setSession(nextSession));
    return () => listener.subscription.unsubscribe();
  }, []);

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

  function addIngredient(value) {
    const normalized = value.trim().toLowerCase();
    if (!normalized || form.ingredients.includes(normalized)) return;
    updateForm("ingredients", [...form.ingredients, normalized]);
    setIngredientInput("");
  }

  function removeIngredient(value) {
    updateForm(
      "ingredients",
      form.ingredients.filter((item) => item !== value),
    );
  }

  async function recommend() {
    setLoading(true);
    setError("");
    setSelectedRecipe(null);
    try {
      const response = await fetch(`${API_BASE_URL}/recommendations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, user_id: userId, limit: 12 }),
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
          context: { cuisine: form.cuisine, diet: form.diet, course: form.course, ingredients: form.ingredients },
        }),
      });
      if (!response.ok) throw new Error(`Feedback could not be saved (${response.status})`);
      setResults((current) =>
        current.map((item) => (item.recipe_id === recipe.recipe_id ? { ...item, lastFeedback: action } : item)),
      );
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
            metadata={metadata}
            ingredientInput={ingredientInput}
            suggestions={suggestions}
            loading={loading}
            onInput={setIngredientInput}
            onAddIngredient={addIngredient}
            onRemoveIngredient={removeIngredient}
            onUpdate={updateForm}
            onSubmit={recommend}
          />
        </section>

        {error && <div className="rounded-lg border border-error/20 bg-red-50 px-4 py-3 text-sm text-error">{error}</div>}

        <section className="grid gap-6 lg:grid-cols-[1fr_420px]">
          <ResultsGrid results={results} loading={loading} onOpen={loadRecipe} onFeedback={sendFeedback} />
          <RecipeDetailPanel recipe={selectedRecipe} loading={detailLoading} />
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
  metadata,
  ingredientInput,
  suggestions,
  loading,
  onInput,
  onAddIngredient,
  onRemoveIngredient,
  onUpdate,
  onSubmit,
}) {
  return (
    <section className="rounded-xl border border-outline-variant bg-surface-container-lowest p-5 shadow-sm">
      <div className="grid gap-4 sm:grid-cols-2">
        <Select label="Cuisine" value={form.cuisine} options={metadata.cuisines} onChange={(value) => onUpdate("cuisine", value)} />
        <Select label="Meal" value={form.course} options={metadata.courses} onChange={(value) => onUpdate("course", value)} />
      </div>

      <div className="mt-5">
        <label className="label">Dietary preference</label>
        <div className="mt-2 flex flex-wrap gap-2">
          {metadata.diets.map((diet) => (
            <button
              key={diet}
              onClick={() => onUpdate("diet", diet)}
              className={`chip ${form.diet === diet ? "chip-selected" : ""}`}
            >
              {diet}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-5">
        <label className="label">What's in your fridge?</label>
        <div className="mt-2 rounded-xl bg-surface-container-low p-3 focus-within:ring-2 focus-within:ring-secondary">
          <div className="mb-2 flex flex-wrap gap-2">
            {form.ingredients.map((ingredient) => (
              <span key={ingredient} className="inline-flex items-center gap-1 rounded-lg bg-primary px-3 py-1 text-sm text-on-primary">
                {ingredient}
                <button onClick={() => onRemoveIngredient(ingredient)} aria-label={`Remove ${ingredient}`}>
                  {icon("close", "text-base")}
                </button>
              </span>
            ))}
          </div>
          <input
            value={ingredientInput}
            onChange={(event) => onInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                onAddIngredient(ingredientInput);
              }
            }}
            className="w-full border-0 bg-transparent p-0 text-on-surface placeholder:text-outline focus:ring-0"
            placeholder="Add ingredients..."
          />
        </div>
        {suggestions.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {suggestions.map((suggestion) => (
              <button key={suggestion} onClick={() => onAddIngredient(suggestion)} className="suggestion-chip">
                {suggestion}
              </button>
            ))}
          </div>
        )}
      </div>

      <button onClick={onSubmit} disabled={loading} className="mt-6 flex w-full items-center justify-center gap-2 rounded-full bg-primary px-6 py-4 font-display text-lg font-semibold text-on-primary shadow-lg transition hover:bg-tertiary disabled:opacity-60">
        {loading ? "Finding recipes..." : "Generate Recipes"}
        {icon("auto_awesome")}
      </button>
    </section>
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
          {options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
        {icon("expand_more", "pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-outline")}
      </span>
    </label>
  );
}

function ResultsGrid({ results, loading, onOpen, onFeedback }) {
  if (loading) {
    return <section className="rounded-xl bg-surface-container-low p-8 text-on-surface-variant">Scoring diet, pantry, cuisine, and feedback...</section>;
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
          <p className="text-sm text-on-surface-variant">Ranked by pantry coverage first, then cuisine and learning signals.</p>
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

function RecipeCard({ recipe, onOpen, onFeedback }) {
  const percent = Math.round(recipe.score.final_score * 100);
  return (
    <article className="overflow-hidden rounded-xl border border-outline-variant bg-surface-container-lowest shadow-sm transition hover:shadow-md">
      <div className="flex h-32 items-end bg-[linear-gradient(135deg,#d8e6d9,#cde5ff)] p-4">
        <div className="rounded-full bg-white/90 px-3 py-1 text-sm font-semibold text-primary">{percent}% Match</div>
      </div>
      <div className="p-4">
        <div className="mb-3 flex flex-wrap gap-2">
          <span className="meta-pill">{recipe.cuisine}</span>
          <span className="meta-pill">{recipe.course}</span>
          <span className="meta-pill">{recipe.diet}</span>
          {recipe.score.alternate_course && <span className="meta-pill-alt">alternate meal</span>}
        </div>
        <h3 className="min-h-16 font-display text-lg font-semibold text-primary">{recipe.name}</h3>
        <ScoreLine label="Pantry match" value={recipe.score.ingredient_coverage} />
        <ScoreLine label="Recipe coverage" value={recipe.score.recipe_coverage} />
        <div className="mt-3 text-xs text-on-surface-variant">
          Uses: {recipe.score.matched_ingredients.length ? recipe.score.matched_ingredients.join(", ") : "No selected ingredients"}
        </div>
        <div className="mt-4 grid grid-cols-[1fr_auto_auto] gap-2">
          <button onClick={() => onOpen(recipe.recipe_id)} className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-on-primary">
            Learn More
          </button>
          <button onClick={() => onFeedback(recipe, "like")} className="small-button" aria-label="Like">
            {icon("thumb_up")}
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

function ScoreLine({ label, value }) {
  return (
    <div className="mt-3">
      <div className="mb-1 flex justify-between text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
        <span>{label}</span>
        <span>{Math.round(value * 100)}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-surface-container-high">
        <div className="h-full rounded-full bg-primary" style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
    </div>
  );
}

function RecipeDetailPanel({ recipe, loading }) {
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
      <div className="mt-4 grid grid-cols-3 gap-2 text-center text-sm">
        <Info label="Prep" value={`${recipe.prep_time_mins ?? 0}m`} />
        <Info label="Cook" value={`${recipe.cook_time_mins ?? 0}m`} />
        <Info label="Serves" value={recipe.servings ?? "-"} />
      </div>
      <h3 className="mt-6 font-display text-lg font-semibold text-primary">Ingredients</h3>
      <div className="mt-3 flex flex-wrap gap-2">
        {recipe.ingredients.map((ingredient) => (
          <span key={ingredient} className="rounded-lg bg-surface-container px-3 py-1 text-sm text-on-surface-variant">
            {ingredient}
          </span>
        ))}
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
