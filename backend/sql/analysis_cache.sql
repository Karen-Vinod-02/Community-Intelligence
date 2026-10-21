create table if not exists public.analysis_cache (
    cache_key text primary key check (length(cache_key) = 64),
    payload jsonb not null,
    expires_at timestamptz not null,
    updated_at timestamptz not null default now()
);

create index if not exists analysis_cache_expires_at_idx
    on public.analysis_cache (expires_at);

alter table public.analysis_cache enable row level security;

-- Cache access is backend-only. Keep browser roles out of this table.
revoke all on table public.analysis_cache from anon, authenticated;
grant all on table public.analysis_cache to service_role;

-- The backend accesses this table with its server-only Supabase service-role key. 
