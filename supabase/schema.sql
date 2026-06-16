create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  created_at timestamptz not null default now()
);

create table if not exists public.recipe_feedback (
  id bigint generated always as identity primary key,
  user_id text not null,
  recipe_id integer not null,
  action text not null check (action in ('like', 'dislike', 'save', 'cooked', 'not_relevant')),
  reason text,
  comment text,
  context jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.saved_recipes (
  id bigint generated always as identity primary key,
  user_id text not null,
  recipe_id integer not null,
  created_at timestamptz not null default now(),
  unique (user_id, recipe_id)
);

create table if not exists public.user_preferences (
  user_id text primary key,
  dietary_preference text,
  pantry_ingredients text[] not null default array[]::text[],
  labels jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.recommendation_events (
  id bigint generated always as identity primary key,
  user_id text,
  request_payload jsonb not null,
  recipe_ids integer[] not null,
  created_at timestamptz not null default now()
);

create table if not exists public.recipe_ranking_overrides (
  id bigint generated always as identity primary key,
  recipe_id integer not null,
  boost numeric not null default 0,
  reason text,
  starts_at timestamptz,
  ends_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists recipe_feedback_recipe_id_idx on public.recipe_feedback(recipe_id);
create index if not exists recipe_feedback_user_id_idx on public.recipe_feedback(user_id);
create index if not exists recommendation_events_user_id_idx on public.recommendation_events(user_id);
create index if not exists user_preferences_dietary_preference_idx on public.user_preferences(dietary_preference);
