# Database migration validation

`database-test/validate_supabase.sh` applies the canonical schema and every
file in `supabase/migrations/` to a disposable local `pgvector/pgvector:pg16`
container. It creates only minimal local `auth` and `storage` stubs needed by
the canonical schema, runs with `ON_ERROR_STOP=1`, checks the director tables,
and verifies that legacy non-tenant RAG RPC overloads are gone.

The script does not read `.env`, use Supabase credentials, or connect to an
external database. The container is forcibly removed on exit. Exit code `2`
means Docker was unavailable and validation was skipped.

`20261010_application_tables.sql` is deliberately a follow-on migration to the
canonical schema: it expects the canonical `vaults` and `media` relationships
to exist. This is checked by the isolated run rather than by touching a live
database.

Run from the repository root:

```sh
./database-test/validate_supabase.sh
```
